use candle_core::{DType, Device, IndexOp, Shape, Tensor};

pub struct Arena {
    buffer: Tensor,
    end_idx: usize,
    capacity: usize,
}

impl Arena {
    pub fn new<S: Into<Shape>>(shape: S, size: usize, dtype: DType, device: &Device) -> Result<Self, candle_core::Error> {
        let shape: Shape = shape.into();
        let mut shape = shape.into_dims();
        shape.insert(0, size);
        let buffer = Tensor::zeros(shape, dtype, device)?;
        Ok(Arena {
            buffer,
            end_idx: 0,
            capacity: size,
        })
    }

    pub fn alloc(&mut self) -> Tensor {
        let index = self.end_idx;
        self.end_idx += 1;
        self.buffer.i(index).unwrap()
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
