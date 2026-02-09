use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, DType, InplaceOp1, Layout, MetalStorage, Tensor};
use half::{bf16, f16};
use std::ptr;
use float8::F8E4M3;

pub trait TensorType {
    fn type_matches(dtype: DType) -> bool;
    const DTYPE: DType;
}

macro_rules! tensor_type {
    ($ty: ty, $dtype:ident) => {
        impl TensorType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn type_matches(dtype: DType) -> bool {
                matches!(dtype, DType::$dtype)
            }
        }
    };
}

tensor_type!(u8, U8);
tensor_type!(u32, U32);
tensor_type!(i16, I16);
tensor_type!(i32, I32);
tensor_type!(i64, I64);
tensor_type!(bf16, BF16);
tensor_type!(f16, F16);
tensor_type!(f32, F32);
tensor_type!(f64, F64);
tensor_type!(F8E4M3, F8E4M3);

struct InplaceCopyOp<'a, T: TensorType> {
    slice: &'a [T],
}

macro_rules! cpu_copy {
        ($storage:ident, $start:ident, $end:ident, $slice:expr, [$($primitive:ident, $dtype:ident),*]) => {
            match $storage {
                $(
                CpuStorage::$dtype(cpu_storage) => {
                    let dst: &mut [$primitive] = &mut cpu_storage.as_mut_slice()[$start..$end];
                    let src_bytes = $slice.as_ptr() as *const u8;
                    let dst_bytes = dst.as_ptr() as *mut u8;
                    let byte_count = std::mem::size_of_val($slice);
                    unsafe {
                        std::ptr::copy_nonoverlapping(src_bytes, dst_bytes, byte_count);
                    }
                }
                )*
                _ => candle_core::bail!("unsupported dtype for inplace copy"),
            }
        }
}

#[cfg(target_os = "linux")]
macro_rules! cuda_copy {
        ($storage:ident, $device:ident, $start:ident, $end:ident, $slice:expr, [$($primitive:ident, $dtype:ident),*]) => {
            match $storage.dtype() {
                $(
                DType::$dtype => {
                    unsafe {
                        let mut dst_slice = $storage.as_cuda_slice_mut::<$primitive>().unwrap().slice_mut($start..$end);
                        let src_slice = std::mem::transmute::<&[T], &[$primitive]>($slice);
                        $device.memcpy_htod(src_slice, &mut dst_slice)
                    }
                }
                )*
                _ => candle_core::bail!("unsupported dtype for inplace copy"),
            }
        }
}

impl<T: TensorType> InplaceOp1 for InplaceCopyOp<'_, T> {
    fn name(&self) -> &'static str {
        "copy"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, layout: &Layout) -> candle_core::Result<()> {
        if !T::type_matches(storage.dtype()) {
            candle_core::bail!("type mismatch for copy operation");
        }
        let (start, end) = layout.contiguous_offsets().expect("operation only supports contiguous offsets");
        let elements_count = end - start;
        if elements_count != self.slice.len() {
            candle_core::bail!("dst elements count {} doesn't match src elements count {} for inplace copy", elements_count, self.slice.len());
        }
        cpu_copy! {
            storage, start, end, self.slice,
                [
                    u8, U8,
                    u32, U32,
                    i16, I16,
                    i32, I32,
                    i64, I64,
                    bf16, BF16,
                    f16, F16,
                    f32, F32,
                    f64, F64,
                    F8E4M3, F8E4M3
                ]
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(&self, storage: &mut MetalStorage, layout: &Layout) -> candle_core::Result<()> {
        use objc2_foundation::NSRange;
        if !T::type_matches(storage.dtype()) {
            candle_core::bail!("type mismatch for copy operation");
        }

        let (start, end) = layout.contiguous_offsets().expect("operation only supports contiguous offsets");
        let elem_count = end - start;

        if self.slice.len() < elem_count {
            return Err(candle_core::Error::msg("Source slice too small"));
        }
        if self.slice.len() > elem_count {
            return Err(candle_core::Error::msg("Source slice too big"));
        }
        let byte_size = elem_count *  storage.dtype().size_in_bytes().min(1);
        unsafe {
            let dst_ptr = storage.buffer().contents() as *mut T;
            let dst_ptr = dst_ptr.add(start);
            ptr::copy_nonoverlapping(self.slice.as_ptr(), dst_ptr, elem_count);
            storage.buffer().did_modify_range(NSRange::new(start * byte_size, elem_count * byte_size));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn cuda_fwd(&self, storage: &mut candle_core::CudaStorage, layout: &Layout) -> candle_core::Result<()> {
        use float8::F8E4M3;

        if !T::type_matches(storage.dtype()) {
            candle_core::bail!("type mismatch for copy operation");
        }

        let (start, end) = layout.contiguous_offsets().expect("operation only supports contiguous offsets");

        let elem_count = end - start;
        if self.slice.len() < elem_count {
            return Err(candle_core::Error::msg("Source slice too small"));
        }
        if self.slice.len() > elem_count {
            return Err(candle_core::Error::msg("Source slice too big"));
        }

        let device = storage.device().clone();
        cuda_copy! {
            storage, device, start, end, self.slice,
            [
                u8, U8,
                u32, U32,
                i16, I16,
                i32, I32,
                i64, I64,
                bf16, BF16,
                F8E4M3, F8E4M3,
                f16, F16,
                f32, F32,
                f64, F64
            ]
        }
    }
}

pub trait InplaceCopy {
    fn inplace_copy<T: TensorType>(&self, slice: &[T]) -> Result<(), candle_core::Error>;
}

impl InplaceCopy for Tensor {
    fn inplace_copy<T: TensorType>(&self, slice: &[T]) -> Result<(), candle_core::Error> {
        self.inplace_op1(&InplaceCopyOp { slice })
    }
}

#[cfg(test)]
mod tests {
    use candle_core::IndexOp;
    use candle_core::Device;

    use super::*;
    macro_rules! successful_test {
        ($name: ident, $ty: ty, $dtype: ident, $device: expr) => {
            #[test]
            fn $name() {
                let device = $device;
                let cpu_device = Device::Cpu;
                let original_tensor = Tensor::zeros((2, 4), DType::$dtype, &device).unwrap();
                let data_to_copy = Tensor::arange(0, 8, &cpu_device)
                    .unwrap()
                    .to_dtype(DType::$dtype)
                    .unwrap()
                    .to_device(&device)
                    .unwrap()
                    .to_vec1::<$ty>()
                    .unwrap();
                original_tensor.i(0).unwrap()
                    .inplace_copy(&data_to_copy.as_slice()[0..4])
                    .unwrap();
                original_tensor.i(1).unwrap()
                    .inplace_copy(&data_to_copy.as_slice()[4..8])
                    .unwrap();
                let actual = original_tensor
                    .to_device(&cpu_device)
                    .unwrap()
                    .reshape(8)
                    .unwrap()
                    .to_vec1::<$ty>()
                    .unwrap();
                let expected = data_to_copy.to_vec();
                assert_eq!(expected, actual);
            }
        };
    }

    successful_test!(successful_inplace_copy_cpu_u8, u8, U8, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_u32, u32, U32, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_i16, i16, I16, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_i32, i32, I32, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_i64, i64, I64, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_bf16, bf16, BF16, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_f16, f16, F16, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_f32, f32, F32, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_f64, f64, F64, Device::Cpu);
    successful_test!(successful_inplace_copy_cpu_f8e4m3, F8E4M3, F8E4M3, Device::Cpu);


    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_u8, u8, U8, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_u32, u32, U32, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_i16, i16, I16, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_i32, i32, I32, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_i64, i64, I64, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_bf16, bf16, BF16, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_f16, f16, F16, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_f32, f32, F32, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_f64, f64, F64, Device::new_metal(0).unwrap());
    #[cfg(target_os = "macos")]
    successful_test!(successful_inplace_copy_metal_f8e4m3, F8E4M3, F8E4M3, Device::new_metal(0).unwrap());

    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_u8, u8, U8, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_u32, u32, U32, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_i16, i16, I16, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_i32, i32, I32, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_i64, i64, I64, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_bf16, bf16, BF16, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_f16, f16, F16, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_f32, f32, F32, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_f64, f64, F64, Device::new_cuda(0).unwrap());
    #[cfg(target_os = "linux")]
    successful_test!(successful_inplace_copy_cuda_f8e4m3, F8E4M3, F8E4M3, Device::new_cuda(0).unwrap());

    macro_rules! inplace_failure_less_elements {
        ($name: ident, $device: expr) => {
            #[test]
            #[should_panic]
            fn $name() {
                let device = $device;
                let original_tensor = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
                let data_to_copy = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
                original_tensor.inplace_copy(&data_to_copy).unwrap();
            }
        }
    }

    inplace_failure_less_elements!(failure_inplace_copy_cpu_if_less_elements, Device::Cpu);
    #[cfg(target_os = "macos")]
    inplace_failure_less_elements!(failure_inplace_copy_metal_if_less_elements, Device::new_metal(0).unwrap());
    #[cfg(target_os = "linux")]
    inplace_failure_less_elements!(failure_inplace_copy_cuda_if_less_elements, Device::new_cuda(0).unwrap());

    macro_rules! inplace_failure_more_elements {
        ($name: ident, $device: expr) => {
            #[test]
            #[should_panic]
            fn $name() {
                let device = $device;
                let original_tensor = Tensor::zeros((2, 4), DType::F32, &device).unwrap();
                let data_to_copy = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
                original_tensor.inplace_copy(&data_to_copy).unwrap();
            }
        }
    }

    inplace_failure_more_elements!(failure_inplace_copy_cpu_if_more_elements, Device::Cpu);
    #[cfg(target_os = "macos")]
    inplace_failure_more_elements!(failure_inplace_copy_metal_if_more_elements, Device::new_metal(0).unwrap());
    #[cfg(target_os = "linux")]
    inplace_failure_more_elements!(failure_inplace_copy_cuda_if_more_elements, Device::new_cuda(0).unwrap());
}
