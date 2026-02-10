use std::marker::PhantomData;

use candle_core::{Device, IndexOp, Shape, Tensor};

use crate::inplace_copy::{InplaceCopy, TensorType};

pub struct Arena<T: TensorType> {
    buffer: Tensor,
    end_idx: usize,
    capacity: usize,
    _phantom: PhantomData<T>,
}

#[derive(Clone, Debug)]
pub struct AllocationError {
    message: String,
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arena allocation failed: {}", self.message)
    }
}

impl<T: TensorType> Arena<T> {
    pub fn new<S: Into<Shape>>(
        shape: S,
        size: usize,
        device: &Device,
    ) -> Result<Self, candle_core::Error> {
        let shape: Shape = shape.into();
        let mut shape = shape.into_dims();
        shape.insert(0, size);
        let buffer = Tensor::zeros(shape, T::DTYPE, device)?;
        Ok(Arena {
            buffer,
            end_idx: 0,
            capacity: size,
            _phantom: PhantomData,
        })
    }

    pub fn alloc(&mut self, slice: &[T]) -> Result<Tensor, AllocationError> {
        let index = self.end_idx;
        if index >= self.capacity {
            return Err(AllocationError {
                message: "capacity exceeded".to_owned(),
            });
        }
        self.end_idx += 1;
        let tensor = self.buffer.i(index).map_err(|_| AllocationError {
            message: "failed to select index".to_owned(),
        })?;
        tensor.inplace_copy(slice).map_err(|_| AllocationError {
            message: "failed to inplace copy to tensor from arena".to_owned(),
        })?;
        Ok(tensor)
    }

    pub fn reset(&mut self) {
        self.end_idx = 0;
    }

    pub fn len(&self) -> usize {
        self.end_idx
    }

    pub fn is_empty(&self) -> bool {
        self.end_idx == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float8::F8E4M3;
    use half::{bf16, f16};

    #[test]
    fn test_basic_allocation_and_data_integrity() {
        let device = Device::Cpu;
        let mut arena = Arena::<f32>::new(3, 10, &device).unwrap();

        let data1 = [1.0_f32, 2.0, 3.0];
        let tensor1 = arena.alloc(&data1).unwrap();

        let data2 = [4.0_f32, 5.0, 6.0];
        let tensor2 = arena.alloc(&data2).unwrap();

        let retrieved1 = tensor1.to_vec1::<f32>().unwrap();
        let retrieved2 = tensor2.to_vec1::<f32>().unwrap();

        assert_eq!(retrieved1, vec![1.0, 2.0, 3.0]);
        assert_eq!(retrieved2, vec![4.0, 5.0, 6.0]);

        assert_eq!(arena.len(), 2);
        assert!(!arena.is_empty());
    }

    #[test]
    fn test_fill_to_capacity() {
        let device = Device::Cpu;
        let capacity = 5;
        let mut arena = Arena::<i32>::new(2, capacity, &device).unwrap();

        for i in 0..capacity {
            let data = [i as i32, (i + 1) as i32];
            let result = arena.alloc(&data);
            assert!(result.is_ok(), "allocation {} should succeed", i);
        }

        assert_eq!(arena.len(), capacity);
        assert_eq!(arena.capacity(), capacity);
    }

    #[test]
    fn test_capacity_exceeded() {
        let device = Device::Cpu;
        let capacity = 3;
        let mut arena = Arena::<f32>::new(1, capacity, &device).unwrap();

        for i in 0..capacity {
            arena.alloc(&[i as f32]).unwrap();
        }

        let result = arena.alloc(&[99.0]);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.message, "capacity exceeded");
        assert!(format!("{}", err).contains("capacity exceeded"));
    }

    #[test]
    fn test_reset_and_reuse() {
        let device = Device::Cpu;
        let mut arena = Arena::<f32>::new(2, 5, &device).unwrap();

        arena.alloc(&[1.0, 2.0]).unwrap();
        arena.alloc(&[3.0, 4.0]).unwrap();
        assert_eq!(arena.len(), 2);

        arena.reset();
        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());

        let tensor = arena.alloc(&[10.0, 20.0]).unwrap();
        assert_eq!(arena.len(), 1);

        let retrieved = tensor.to_vec1::<f32>().unwrap();
        assert_eq!(retrieved, vec![10.0, 20.0]);
    }

    #[test]
    fn test_empty_arena() {
        let device = Device::Cpu;
        let arena = Arena::<f32>::new(1, 10, &device).unwrap();

        assert_eq!(arena.len(), 0);
        assert!(arena.is_empty());
        assert_eq!(arena.capacity(), 10);
    }

    #[test]
    fn test_single_capacity() {
        let device = Device::Cpu;
        let mut arena = Arena::<i64>::new(4, 1, &device).unwrap();

        let data = [100_i64, 200, 300, 400];
        let tensor = arena.alloc(&data).unwrap();
        assert_eq!(arena.len(), 1);

        let retrieved = tensor.to_vec1::<i64>().unwrap();
        assert_eq!(retrieved, vec![100, 200, 300, 400]);

        assert!(arena.alloc(&[1, 2, 3, 4]).is_err());
    }

    #[test]
    fn test_multidimensional_shapes() {
        let device = Device::Cpu;

        let mut arena = Arena::<f32>::new((3, 4), 5, &device).unwrap();
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = arena.alloc(&data).unwrap();

        assert_eq!(tensor.dims(), &[3, 4]);
        let retrieved = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(retrieved, data);
    }

    macro_rules! test_dtype {
        ($name:ident, $ty:ty, $dtype:ident, $values:expr) => {
            #[test]
            fn $name() {
                let device = Device::Cpu;
                let mut arena = Arena::<$ty>::new(4, 10, &device).unwrap();

                let data: [$ty; 4] = $values;
                let tensor = arena.alloc(&data).unwrap();

                let retrieved = tensor.to_vec1::<$ty>().unwrap();
                assert_eq!(retrieved, data.to_vec());

                assert_eq!(arena.len(), 1);
                assert!(!arena.is_empty());
            }
        };
    }

    test_dtype!(test_arena_u8, u8, U8, [1, 2, 3, 4]);
    test_dtype!(test_arena_u32, u32, U32, [100, 200, 300, 400]);
    test_dtype!(test_arena_i16, i16, I16, [-10, 20, -30, 40]);
    test_dtype!(test_arena_i32, i32, I32, [-1000, 2000, -3000, 4000]);
    test_dtype!(test_arena_i64, i64, I64, [-100000, 200000, -300000, 400000]);
    test_dtype!(test_arena_f32, f32, F32, [1.5, 2.5, 3.5, 4.5]);
    test_dtype!(test_arena_f64, f64, F64, [1.1, 2.2, 3.3, 4.4]);

    #[test]
    fn test_arena_bf16() {
        let device = Device::Cpu;
        let mut arena = Arena::<bf16>::new(4, 10, &device).unwrap();

        let data = [
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
        ];
        let tensor = arena.alloc(&data).unwrap();

        let retrieved = tensor.to_vec1::<bf16>().unwrap();
        assert_eq!(retrieved.len(), 4);
    }

    #[test]
    fn test_arena_f16() {
        let device = Device::Cpu;
        let mut arena = Arena::<f16>::new(4, 10, &device).unwrap();

        let data = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
        ];
        let tensor = arena.alloc(&data).unwrap();

        let retrieved = tensor.to_vec1::<f16>().unwrap();
        assert_eq!(retrieved.len(), 4);
    }

    #[test]
    fn test_arena_f8e4m3() {
        let device = Device::Cpu;
        let mut arena = Arena::<F8E4M3>::new(4, 10, &device).unwrap();

        let data = [F8E4M3::ONE, F8E4M3::ONE, F8E4M3::ONE, F8E4M3::ONE];
        let tensor = arena.alloc(&data).unwrap();

        let retrieved = tensor.to_vec1::<F8E4M3>().unwrap();
        assert_eq!(retrieved.len(), 4);
    }

    #[test]
    fn test_allocation_independence() {
        let device = Device::Cpu;
        let mut arena = Arena::<f32>::new(3, 10, &device).unwrap();

        let mut tensors = Vec::new();

        for i in 0..5 {
            let base = i as f32 * 10.0;
            let data = [base, base + 1.0, base + 2.0];
            tensors.push(arena.alloc(&data).unwrap());
        }

        for (i, tensor) in tensors.iter().enumerate() {
            let retrieved = tensor.to_vec1::<f32>().unwrap();
            let base = i as f32 * 10.0;
            assert_eq!(retrieved, vec![base, base + 1.0, base + 2.0]);
        }
    }

    #[test]
    fn test_reset_when_full() {
        let device = Device::Cpu;
        let capacity = 3;
        let mut arena = Arena::<i32>::new(1, capacity, &device).unwrap();

        for i in 0..capacity {
            arena.alloc(&[i as i32]).unwrap();
        }

        assert_eq!(arena.len(), capacity);
        assert!(arena.alloc(&[999]).is_err());

        arena.reset();
        assert_eq!(arena.len(), 0);

        let tensor = arena.alloc(&[42]).unwrap();
        let retrieved = tensor.to_vec1::<i32>().unwrap();
        assert_eq!(retrieved, vec![42]);
    }

    #[test]
    fn test_allocation_error_display() {
        let error = AllocationError {
            message: "test error".to_owned(),
        };

        let display = format!("{}", error);
        assert!(display.contains("arena allocation failed"));
        assert!(display.contains("test error"));
    }

    #[test]
    fn test_scalar_tensors() {
        let device = Device::Cpu;
        let mut arena = Arena::<f32>::new((), 5, &device).unwrap();

        let t1 = arena.alloc(&[3.11]).unwrap();
        let t2 = arena.alloc(&[2.71]).unwrap();

        let empty: &[usize] = &[];
        assert_eq!(t1.dims(), empty);
        assert_eq!(t2.dims(), empty);

        let v1 = t1.to_vec0::<f32>().unwrap();
        let v2 = t2.to_vec0::<f32>().unwrap();

        assert_eq!(v1, 3.11);
        assert_eq!(v2, 2.71);
    }

    #[test]
    fn test_large_allocations() {
        let device = Device::Cpu;
        let size = 1000;
        let mut arena = Arena::<f32>::new(size, 10, &device).unwrap();

        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let tensor = arena.alloc(&data).unwrap();

        let retrieved = tensor.to_vec1::<f32>().unwrap();
        assert_eq!(retrieved.len(), size);
        assert_eq!(retrieved, data);
    }
}
