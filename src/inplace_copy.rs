use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, DType, InplaceOp1, Layout, MetalStorage, Tensor};
use half::{bf16, f16};
use std::ptr;

pub trait TensorType {
    type CpuRepresentation;
    fn matches_cpu_storage(storage: &CpuStorage) -> bool;
    fn matches_metal_storage(storage: &MetalStorage) -> bool;
}

impl TensorType for bf16 {
    type CpuRepresentation = bf16;

    fn matches_cpu_storage(storage: &CpuStorage) -> bool {
        matches!(storage, CpuStorage::F16(_))
    }

    fn matches_metal_storage(storage: &MetalStorage) -> bool {
        matches!(storage.dtype(), DType::BF16)
    }
}
impl TensorType for f16 {
    type CpuRepresentation = f16;

    fn matches_cpu_storage(storage: &CpuStorage) -> bool {
        matches!(storage, CpuStorage::F16(_))
    }

    fn matches_metal_storage(storage: &MetalStorage) -> bool {
        matches!(storage.dtype(), DType::F16)
    }
}

impl TensorType for f32 {
    type CpuRepresentation = f32;

    fn matches_cpu_storage(storage: &CpuStorage) -> bool {
        matches!(storage, CpuStorage::F32(_))
    }

    fn matches_metal_storage(storage: &MetalStorage) -> bool {
        matches!(storage.dtype(), DType::F32)
    }
}

impl TensorType for f64 {
    type CpuRepresentation = f64;

    fn matches_cpu_storage(storage: &CpuStorage) -> bool {
        matches!(storage, CpuStorage::F64(_))
    }

    fn matches_metal_storage(storage: &MetalStorage) -> bool {
        matches!(storage.dtype(), DType::F64)
    }
}

struct InplaceCopyOp<'a, T: TensorType> {
    slice: &'a [T],
}

impl<'a, T: TensorType> InplaceOp1 for InplaceCopyOp<'a, T> {
    fn name(&self) -> &'static str {
        "copy"
    }

    fn cpu_fwd(&self, storage: &mut CpuStorage, _: &Layout) -> candle_core::Result<()> {
        if !T::matches_cpu_storage(storage) {
            candle_core::bail!("type mismatch for copy operation");
        }

        match storage {
            CpuStorage::BF16(s) => {
                let dst: &mut [bf16] = s.as_mut_slice();
                unsafe {
                    let src = std::mem::transmute::<&[T], &[bf16]>(self.slice);
                    dst.copy_from_slice(src);
                }
            }
            CpuStorage::F16(s) => {
                let dst: &mut [f16] = s.as_mut_slice();
                unsafe {
                    let src = std::mem::transmute::<&[T], &[f16]>(self.slice);
                    dst.copy_from_slice(src);
                }
            }
            CpuStorage::F32(s) => {
                let dst: &mut [f32] = s.as_mut_slice();
                unsafe {
                    let src = std::mem::transmute::<&[T], &[f32]>(self.slice);
                    dst.copy_from_slice(src);
                }
            }
            CpuStorage::F64(s) => {
                let dst: &mut [f64] = s.as_mut_slice();
                unsafe {
                    let src = std::mem::transmute::<&[T], &[f64]>(self.slice);
                    dst.copy_from_slice(src);
                }
            }
            _ => candle_core::bail!("unsupported dtype for inplace elu"),
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn metal_fwd(&self, storage: &mut MetalStorage, layout: &Layout) -> candle_core::Result<()> {
        use objc2_foundation::NSRange;
        if !T::matches_metal_storage(storage) {
            candle_core::bail!("type mismatch for copy operation");
        }

        let elem_count = layout.shape().elem_count();
        if self.slice.len() < elem_count {
            return Err(candle_core::Error::msg("Source slice too small"));
        }
        if !layout.is_contiguous() {
            return Err(candle_core::Error::msg("Contiguous layout required"));
        }
        let byte_len = elem_count * 4; // f32 = 4 bytes
        unsafe {
            let dst_ptr = storage.buffer().contents() as *mut T;
            ptr::copy_nonoverlapping(self.slice.as_ptr(), dst_ptr, elem_count);
            storage.buffer().did_modify_range(NSRange::new(0, byte_len));
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> candle_core::Result<()> {
        use float8::F8E4M3;
        let device = storage.device().clone();
        match storage.dtype() {
            DType::U8 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<u8>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[u8]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::U32 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<u32>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[u32]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::I16 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<i32>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[i32]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::I32 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<i32>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[i32]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::I64 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<i32>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[i32]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::BF16 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<bf16>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[bf16]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::F16 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<f16>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[f16]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::F32 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<f32>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[f32]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::F64 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<f64>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[f64]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            DType::F8E4M3 => unsafe {
                let dst_slice = storage.as_cuda_slice_mut::<F8E4M3>().unwrap();
                let src_slice = std::mem::transmute::<&[T], &[F8E4M3]>(self.slice);
                device.memcpy_htod(src_slice, dst_slice)
            },
            _ => candle_core::bail!("dtype not supported!"),
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

