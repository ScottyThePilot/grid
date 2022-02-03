use std::collections::VecDeque;
use std::collections::vec_deque::Iter as VecDequeIter;
use std::collections::vec_deque::IterMut as VecDequeIterMut;
use std::collections::vec_deque::IntoIter as VecDequeIntoIter;
use std::iter::{FlatMap, FusedIterator, Enumerate};



#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseGrid<T> {
  rows: VecDeque<VecDeque<T>>,
  columns: usize
}

impl<T> BaseGrid<T> {
  pub fn new(value: T) -> Self {
    BaseGrid {
      rows: VecDeque::from(vec![
        VecDeque::from(vec![
          value
        ])
      ]),
      columns: 1
    }
  }

  #[inline(always)]
  pub fn width(&self) -> usize {
    self.columns
  }

  #[inline(always)]
  pub fn height(&self) -> usize {
    self.rows.len()
  }

  #[inline]
  fn within(&self, x: usize, y: usize) -> bool {
    x < self.width() && y < self.height()
  }

  #[inline]
  pub fn get(&self, x: usize, y: usize) -> Option<&T> {
    if self.within(x, y) {
      Some(&self.rows[y][x])
    } else {
      None
    }
  }

  #[inline]
  pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
    if self.within(x, y) {
      Some(&mut self.rows[y][x])
    } else {
      None
    }
  }

  #[inline]
  pub fn set(&mut self, x: usize, y: usize, value: T) -> bool {
    if self.within(x, y) {
      self.rows[y][x] = value;
      true
    } else {
      false
    }
  }

  /// Adds a new row to the back (positive y) of the grid
  pub fn push_row_back(&mut self)
  where T: Default {
    self.rows.push_back(empty_row(self.width()));
  }

  /// Adds a new row to the front (negative y) of the grid
  pub fn push_row_front(&mut self)
  where T: Default {
    self.rows.push_front(empty_row(self.width()));
  }

  /// Adds a new column to the back (positive x) of the grid
  pub fn push_column_back(&mut self)
  where T: Default {
    self.columns += 1;
    for row in self.rows.iter_mut() {
      row.push_back(T::default());
    };
  }

  /// Adds a new column to the front (negative x) of the grid
  pub fn push_column_front(&mut self)
  where T: Default {
    self.columns += 1;
    for row in self.rows.iter_mut() {
      row.push_front(T::default());
    };
  }

  #[inline]
  pub fn iter(&self) -> Iter<T> {
    self.into_iter()
  }

  #[inline]
  pub fn iter_mut(&mut self) -> IterMut<T> {
    self.into_iter()
  }

  pub fn cells(&self) -> Cells<T> {
    #[inline]
    fn f<T>(item: (usize, &VecDeque<T>)) -> EnumerateExtra<VecDequeIter<T>> {
      EnumerateExtra::new(item.0, item.1.iter())
    }

    let inner = self.rows.iter()
      .enumerate().flat_map(f as _);
    Cells { inner }
  }

  pub fn cells_mut(&mut self) -> CellsMut<T> {
    #[inline]
    fn f<T>(item: (usize, &mut VecDeque<T>)) -> EnumerateExtra<VecDequeIterMut<T>> {
      EnumerateExtra::new(item.0, item.1.iter_mut())
    }

    let inner = self.rows.iter_mut()
      .enumerate().flat_map(f as _);
    CellsMut { inner }
  }

  pub fn into_cells(self) -> IntoCells<T> {
    #[inline]
    fn f<T>(item: (usize, VecDeque<T>)) -> EnumerateExtra<VecDequeIntoIter<T>> {
      EnumerateExtra::new(item.0, item.1.into_iter())
    }

    let inner = self.rows.into_iter()
      .enumerate().flat_map(f as _);
    IntoCells { inner }
  }
}

impl<T: Default> Default for BaseGrid<T> {
  #[inline]
  fn default() -> Self {
    BaseGrid::new(T::default())
  }
}

impl<'a, T> IntoIterator for &'a BaseGrid<T> {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    let inner = self.rows.iter()
      .flat_map(VecDeque::iter as _);
    Iter { inner }
  }
}

impl<'a, T> IntoIterator for &'a mut BaseGrid<T> {
  type Item = &'a mut T;
  type IntoIter = IterMut<'a, T>;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    let inner = self.rows.iter_mut()
      .flat_map(VecDeque::iter_mut as _);
    IterMut { inner }
  }
}

impl<T> IntoIterator for BaseGrid<T> {
  type Item = T;
  type IntoIter = IntoIter<T>;

  #[inline]
  fn into_iter(self) -> Self::IntoIter {
    let inner = self.rows.into_iter()
      .flat_map(VecDeque::into_iter as _);
    IntoIter { inner }
  }
}



macro_rules! impl_iterator_methods {
  () => {
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
      self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
      self.inner.size_hint()
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where F: FnMut(B, Self::Item) -> B, {
      self.inner.fold(init, f)
    }
  };
}

macro_rules! impl_double_ended_iterator_methods {
  () => {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
      self.inner.next_back()
    }

    #[inline]
    fn rfold<A, F>(self, init: A, f: F) -> A
    where F: FnMut(A, Self::Item) -> A {
      self.inner.rfold(init, f)
    }
  };
}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
  inner: FlatMap<
    VecDequeIter<'a, VecDeque<T>>,
    VecDequeIter<'a, T>,
    for<'r> fn(&'r VecDeque<T>) -> VecDequeIter<'r, T>
  >
}

impl<'a, T> Iterator for Iter<'a, T> {
  type Item = &'a T;

  impl_iterator_methods!();
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
  impl_double_ended_iterator_methods!();
}

impl<'a, T> FusedIterator for Iter<'a, T> {}



#[repr(transparent)]
#[derive(Debug)]
pub struct IterMut<'a, T> {
  inner: FlatMap<
    VecDequeIterMut<'a, VecDeque<T>>,
    VecDequeIterMut<'a, T>,
    for<'r> fn(&'r mut VecDeque<T>) -> VecDequeIterMut<'r, T>
  >
}

impl<'a, T> Iterator for IterMut<'a, T> {
  type Item = &'a mut T;

  impl_iterator_methods!();
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
  impl_double_ended_iterator_methods!();
}

impl<'a, T> FusedIterator for IterMut<'a, T> {}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct IntoIter<T> {
  inner: FlatMap<
    VecDequeIntoIter<VecDeque<T>>,
    VecDequeIntoIter<T>,
    fn(VecDeque<T>) -> VecDequeIntoIter<T>,
  >
}

impl<T> Iterator for IntoIter<T> {
  type Item = T;

  impl_iterator_methods!();
}

impl<T> DoubleEndedIterator for IntoIter<T> {
  impl_double_ended_iterator_methods!();
}

impl<T> FusedIterator for IntoIter<T> {}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Cells<'a, T> {
  inner: FlatMap<
    Enumerate<VecDequeIter<'a, VecDeque<T>>>,
    EnumerateExtra<VecDequeIter<'a, T>>,
    for<'r> fn((usize, &'r VecDeque<T>)) -> EnumerateExtra<VecDequeIter<'r, T>>
  >
}

impl<'a, T> Iterator for Cells<'a, T> {
  type Item = ([usize; 2], &'a T);

  impl_iterator_methods!();
}

impl<'a, T> DoubleEndedIterator for Cells<'a, T> {
  impl_double_ended_iterator_methods!();
}

impl<'a, T> FusedIterator for Cells<'a, T> {}



#[repr(transparent)]
#[derive(Debug)]
pub struct CellsMut<'a, T> {
  inner: FlatMap<
    Enumerate<VecDequeIterMut<'a, VecDeque<T>>>,
    EnumerateExtra<VecDequeIterMut<'a, T>>,
    for<'r> fn((usize, &'r mut VecDeque<T>)) -> EnumerateExtra<VecDequeIterMut<'r, T>>
  >
}

impl<'a, T> Iterator for CellsMut<'a, T> {
  type Item = ([usize; 2], &'a mut T);

  impl_iterator_methods!();
}

impl<'a, T> DoubleEndedIterator for CellsMut<'a, T> {
  impl_double_ended_iterator_methods!();
}

impl<'a, T> FusedIterator for CellsMut<'a, T> {}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct IntoCells<T> {
  inner: FlatMap<
    Enumerate<VecDequeIntoIter<VecDeque<T>>>,
    EnumerateExtra<VecDequeIntoIter<T>>,
    fn((usize, VecDeque<T>)) -> EnumerateExtra<VecDequeIntoIter<T>>
  >
}

impl<T> Iterator for IntoCells<T> {
  type Item = ([usize; 2], T);

  impl_iterator_methods!();
}

impl<T> DoubleEndedIterator for IntoCells<T> {
  impl_double_ended_iterator_methods!();
}

impl<T> FusedIterator for IntoCells<T> {}



#[inline]
fn empty_row<T: Default>(len: usize) -> VecDeque<T> {
  let mut out = Vec::with_capacity(len);
  for _ in 0..len { out.push(T::default()) };
  VecDeque::from(out)
}



macro_rules! map {
  ($self:ident, $expr:expr) => {
    match $expr {
      Some((other_index, value)) => {
        Some(([other_index, $self.index], value))
      },
      None => None
    }
  };
}



#[derive(Debug, Clone)]
pub struct EnumerateExtra<I> {
  index: usize,
  iter: Enumerate<I>
}

impl<I> EnumerateExtra<I> {
  pub fn new<T>(index: usize, iter: I) -> Self
  where I: Iterator<Item = T> {
    EnumerateExtra {
      index,
      iter: iter.enumerate()
    }
  }
}

impl<I, T> Iterator for EnumerateExtra<I>
where I: Iterator<Item = T> {
  type Item = ([usize; 2], T);

  #[inline]
  fn next(&mut self) -> Option<Self::Item> {
    map!(self, self.iter.next())
  }

  #[inline]
  fn nth(&mut self, n: usize) -> Option<Self::Item> {
    map!(self, self.iter.nth(n))
  }

  #[inline]
  fn size_hint(&self) -> (usize, Option<usize>) {
    self.iter.size_hint()
  }

  #[inline]
  fn fold<B, F>(self, init: B, f: F) -> B
  where F: FnMut(B, Self::Item) -> B, {
    self.iter.fold(init, wrap_fold_fn(self.index, f))
  }
}

impl<I, T> DoubleEndedIterator for EnumerateExtra<I>
where I: DoubleEndedIterator<Item = T> + ExactSizeIterator {
  #[inline]
  fn next_back(&mut self) -> Option<Self::Item> {
    map!(self, self.iter.next_back())
  }

  #[inline]
  fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
    map!(self, self.iter.nth_back(n))
  }

  #[inline]
  fn rfold<A, F>(self, init: A, f: F) -> A
  where F: FnMut(A, Self::Item) -> A {
    self.iter.rfold(init, wrap_fold_fn(self.index, f))
  }
}

impl<I> FusedIterator for EnumerateExtra<I> where I: FusedIterator {}



fn wrap_fold_fn<T, A, F>(index: usize, mut f: F) -> impl FnMut(A, (usize, T)) -> A
where F: FnMut(A, ([usize; 2], T)) -> A {
  move |acc, (other_index, value)| {
    f(acc, ([index, other_index], value))
  }
}



#[cfg(test)]
mod tests {
  use super::*;

  use glam::UVec2;

  #[test]
  fn basegrid_cells_indexes() {
    let mut grid = BaseGrid::<UVec2>::default();

    // Expands the grid 3 times to make it 4 high and 4 wide
    for _ in 0..3 {
      grid.push_row_back();
      grid.push_column_back();
    };

    for x in 0..4 {
      for y in 0..4 {
        let pos = UVec2::new(x as u32, y as u32);
        assert!(grid.set(x, y, pos));
      };
    };

    for (pos, value) in grid.into_cells() {
      let value = [value[0] as usize, value[1] as usize];
      assert_eq!(pos, value);
    };
  }
}
