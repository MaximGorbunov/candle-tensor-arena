use std::marker::PhantomData;

use candle_core::{DType, Device, IndexOp, Shape, Tensor};

use crate::inplace_copy::{InplaceCopy, TensorType};

pub struct Arena<T: TensorType> {
    buffer: Tensor,
    end_idx: usize,
    capacity: usize,
    _phantom: PhantomData<T>,
}

#[derive(Clone, Debug)]
pub struct AllocationError;

impl<T: TensorType> Arena<T> {
    pub fn new<S: Into<Shape>>(
        shape: S,
        size: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, candle_core::Error> {
        let shape: Shape = shape.into();
        let mut shape = shape.into_dims();
        shape.insert(0, size);
        let buffer = Tensor::zeros(shape, dtype, device)?;
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
            return Err(AllocationError);
        }
        self.end_idx += 1;
        let tensor = self.buffer.i(index).unwrap();
        tensor
            .inplace_copy(slice)
            .expect("failed to inplacy copy to tensor from arena");
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

    #[test]
    fn test_allocation() {
        let device = Device::Cpu;
        let mut arena = Arena::<f32>::new(1, 100, DType::F32, &device).unwrap();
        for i in 0..100 {
            let t = arena.alloc(&[i as f32]).unwrap();
        }
    }
}
