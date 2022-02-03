use crate::basegrid::BaseGrid;
use crate::basegrid::{Iter as BaseIter, IterMut as BaseIterMut, IntoIter as BaseIntoIter};
use crate::basegrid::{Cells as BaseCells, CellsMut as BaseCellsMut, IntoCells as BaseIntoCells};

use glam::{IVec2, XY};

use std::iter::{FilterMap, FusedIterator};
use std::ops::{Index, IndexMut};



macro_rules! try_bool {
  ($expr:expr) => {
    match $expr {
      Some(value) => value,
      None => return false
    }
  };
}



fn inv_offset_pos(offset: XY<isize>, pos: [usize; 2]) -> IVec2 {
  let x = (pos[0] as isize).checked_add(offset.x).expect("integer overflow");
  let y = (pos[1] as isize).checked_add(offset.y).expect("integer overflow");
  IVec2::new(x as i32, y as i32)
}

fn offset_pos(offset: XY<isize>, pos: IVec2) -> (isize, isize) {
  let x = (pos.x as isize).checked_sub(offset.x).expect("integer overflow");
  let y = (pos.y as isize).checked_sub(offset.y).expect("integer overflow");
  (x, y)
}

#[derive(Debug, Clone)]
struct SparseGridInner<T> {
  base: BaseGrid<Option<T>>,
  offset: XY<isize>
}

impl<T> SparseGridInner<T> {
  fn new(value: T, pos: IVec2) -> Self {
    SparseGridInner {
      base: BaseGrid::new(Some(value)),
      offset: XY {
        x: pos.x as isize,
        y: pos.y as isize
      }
    }
  }

  fn get_pos(&self, pos: IVec2) -> Option<(usize, usize)> {
    let (x, y) = offset_pos(self.offset, pos);
    if x >= 0 && y >= 0 {
      Some((x as usize, y as usize))
    } else {
      None
    }
  }

  fn get_oob(&self, pos: IVec2) -> (isize, isize) {
    let (x, y) = offset_pos(self.offset, pos);
    let width = self.base.width() as isize;
    let height = self.base.height() as isize;
    let x = if x < 0 { x } else if x >= width { x - width + 1 } else { 0 };
    let y = if y < 0 { y } else if y >= height { y - height + 1 } else { 0 };
    (x, y)
  }

  #[inline]
  fn min(&self) -> IVec2 {
    let XY { x, y } = self.offset;
    IVec2::new(x as i32, y as i32)
  }

  #[inline]
  fn max(&self) -> IVec2 {
    let x = self.base.width() as isize + self.offset.x - 1;
    let y = self.base.height() as isize + self.offset.y - 1;
    IVec2::new(x as i32, y as i32)
  }

  #[inline]
  fn push_columns_back(&mut self, c: usize) {
    for _ in 0..c {
      self.base.push_column_back();
    };
  }

  #[inline]
  fn push_columns_front(&mut self, c: usize) {
    for _ in 0..c {
      self.offset.x -= 1;
      self.base.push_column_front();
    };
  }

  #[inline]
  fn push_rows_back(&mut self, c: usize) {
    for _ in 0..c {
      self.base.push_row_back();
    };
  }

  #[inline]
  fn push_rows_front(&mut self, c: usize) {
    for _ in 0..c {
      self.offset.y -= 1;
      self.base.push_row_front();
    };
  }
}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct SparseGrid<T> {
  inner: Option<SparseGridInner<T>>
}

impl<T> SparseGrid<T> {
  #[inline]
  pub fn new() -> Self {
    SparseGrid { inner: None }
  }

  #[inline]
  pub fn width(&self) -> usize {
    match self.inner {
      Some(ref inner) => inner.base.width(),
      None => 0
    }
  }

  #[inline]
  pub fn height(&self) -> usize {
    match self.inner {
      Some(ref inner) => inner.base.height(),
      None => 0
    }
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.inner.is_none()
  }

  #[inline]
  pub fn contains(&self, pos: IVec2) -> bool {
    self.get(pos).is_some()
  }

  pub fn get(&self, pos: IVec2) -> Option<&T> {
    let inner = self.inner.as_ref()?;
    let (x, y) = inner.get_pos(pos)?;
    match inner.base.get(x, y) {
      Some(Some(value)) => Some(value),
      _ => None
    }
  }

  pub fn get_mut(&mut self, pos: IVec2) -> Option<&mut T> {
    let inner = self.inner.as_mut()?;
    let (x, y) = inner.get_pos(pos)?;
    match inner.base.get_mut(x, y) {
      Some(Some(value)) => Some(value),
      _ => None
    }
  }

  pub fn set(&mut self, pos: IVec2, value: T) -> bool {
    let inner = try_bool!(self.inner.as_mut());
    let (x, y) = try_bool!(inner.get_pos(pos));
    inner.base.set(x, y, Some(value))
  }

  /// Puts a value into the grid at the given position,
  /// expanding the underlying structure to fit it if neccessary.
  pub fn put(&mut self, pos: IVec2, value: T) {
    if let Some(ref mut inner) = self.inner {
      let (x_oob, y_oob) = inner.get_oob(pos);

      match x_oob.signum() {
        0 => (),
        1 => inner.push_columns_back(x_oob as usize),
        -1 => inner.push_columns_front(-x_oob as usize),
        _ => unreachable!()
      };

      match y_oob.signum() {
        0 => (),
        1 => inner.push_rows_back(y_oob as usize),
        -1 => inner.push_rows_front(-y_oob as usize),
        _ => unreachable!()
      };

      match inner.get_pos(pos) {
        Some((x, y)) => assert!(inner.base.set(x, y, Some(value))),
        None => unreachable!()
      };
    } else {
      self.inner = Some(SparseGridInner::new(value, pos));
    };
  }

  #[inline]
  pub fn min(&self) -> Option<IVec2> {
    match self.inner {
      Some(ref inner) => Some(inner.min()),
      None => None
    }
  }

  #[inline]
  pub fn max(&self) -> Option<IVec2> {
    match self.inner {
      Some(ref inner) => Some(inner.max()),
      None => None
    }
  }

  #[inline]
  pub fn iter(&self) -> Iter<T> {
    self.into_iter()
  }

  #[inline]
  pub fn iter_mut(&mut self) -> IterMut<T> {
    self.into_iter()
  }

  #[inline]
  pub fn cells(&self) -> Cells<T> {
    Cells {
      inner: match self.inner {
        Some(ref inner) => Some({
          (inner.offset, inner.base.cells())
        }),
        None => None
      }
    }
  }

  #[inline]
  pub fn cells_mut(&mut self) -> CellsMut<T> {
    CellsMut {
      inner: match self.inner {
        Some(ref mut inner) => Some({
          (inner.offset, inner.base.cells_mut())
        }),
        None => None
      }
    }
  }

  #[inline]
  pub fn into_cells(self) -> IntoCells<T> {
    IntoCells {
      inner: match self.inner {
        Some(inner) => Some({
          (inner.offset, inner.base.into_cells())
        }),
        None => None
      }
    }
  }
}

impl<T> Default for SparseGrid<T> {
  #[inline]
  fn default() -> Self {
    Self::new()
  }
}

impl<T> Index<IVec2> for SparseGrid<T> {
  type Output = T;

  #[inline]
  fn index(&self, pos: IVec2) -> &Self::Output {
    self.get(pos).expect("index out of bounds")
  }
}

impl<T> IndexMut<IVec2> for SparseGrid<T> {
  #[inline]
  fn index_mut(&mut self, pos: IVec2) -> &mut Self::Output {
    self.get_mut(pos).expect("index out of bounds")
  }
}

impl<T> FromIterator<(IVec2, T)> for SparseGrid<T> {
  fn from_iter<I: IntoIterator<Item = (IVec2, T)>>(iter: I) -> Self {
    let mut grid = SparseGrid::new();
    for (pos, value) in iter {
      grid.put(pos, value);
    };

    grid
  }
}

impl<'a, T> IntoIterator for &'a SparseGrid<T> {
  type Item = &'a T;
  type IntoIter = Iter<'a, T>;

  fn into_iter(self) -> Self::IntoIter {
    Iter {
      inner: match self.inner {
        Some(ref inner) => Some({
          inner.base.iter()
            .filter_map(Option::as_ref as _)
        }),
        None => None
      }
    }
  }
}

impl<'a, T> IntoIterator for &'a mut SparseGrid<T> {
  type Item = &'a mut T;
  type IntoIter = IterMut<'a, T>;

  fn into_iter(self) -> Self::IntoIter {
    IterMut {
      inner: match self.inner {
        Some(ref mut inner) => Some({
          inner.base.iter_mut()
            .filter_map(Option::as_mut as _)
        }),
        None => None
      }
    }
  }
}

impl<T> IntoIterator for SparseGrid<T> {
  type Item = T;
  type IntoIter = IntoIter<T>;

  fn into_iter(self) -> Self::IntoIter {
    IntoIter {
      inner: match self.inner {
        Some(inner) => Some({
          inner.base.into_iter()
            .filter_map(|v| { v } as _)
        }),
        None => None
      }
    }
  }
}



macro_rules! impl_iterator_methods {
  () => {
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
      match self.inner {
        Some(ref mut inner) => inner.next(),
        None => None
      }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
      match self.inner {
        Some(ref inner) => inner.size_hint(),
        None => (0, Some(0))
      }
    }

    #[inline]
    fn fold<B, F>(self, init: B, f: F) -> B
    where F: FnMut(B, Self::Item) -> B, {
      match self.inner {
        Some(inner) => inner.fold(init, f),
        None => init
      }
    }
  };
}

macro_rules! impl_double_ended_iterator_methods {
  () => {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
      match self.inner {
        Some(ref mut inner) => inner.next_back(),
        None => None
      }
    }

    #[inline]
    fn rfold<B, F>(self, init: B, f: F) -> B
    where F: FnMut(B, Self::Item) -> B, {
      match self.inner {
        Some(inner) => inner.rfold(init, f),
        None => init
      }
    }
  };
}



#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Iter<'a, T> {
  inner: Option<FilterMap<
    BaseIter<'a, Option<T>>,
    for<'r> fn(&'r Option<T>) -> Option<&'r T>
  >>
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
  inner: Option<FilterMap<
    BaseIterMut<'a, Option<T>>,
    for<'r> fn(&'r mut Option<T>) -> Option<&'r mut T>
  >>
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
  inner: Option<FilterMap<
    BaseIntoIter<Option<T>>,
    fn(Option<T>) -> Option<T>
  >>
}

impl<T> Iterator for IntoIter<T> {
  type Item = T;

  impl_iterator_methods!();
}

impl<T> DoubleEndedIterator for IntoIter<T> {
  impl_double_ended_iterator_methods!();
}

impl<T> FusedIterator for IntoIter<T> {}



macro_rules! impl_iterator_next_cells {
  () => {
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
      match self.inner {
        Some((offset, ref mut inner)) => inner.find_map(|(pos, value)| match value {
          Some(value) => Some((inv_offset_pos(offset, pos), value)),
          None => None
        }),
        None => None
      }
    }
  };
}



pub struct Cells<'a, T> {
  inner: Option<(XY<isize>, BaseCells<'a, Option<T>>)>
}

impl<'a, T> Iterator for Cells<'a, T> {
  type Item = (IVec2, &'a T);

  impl_iterator_next_cells!();
}

impl<'a, T> FusedIterator for Cells<'a, T> {}



pub struct CellsMut<'a, T> {
  inner: Option<(XY<isize>, BaseCellsMut<'a, Option<T>>)>
}

impl<'a, T> Iterator for CellsMut<'a, T> {
  type Item = (IVec2, &'a mut T);

  impl_iterator_next_cells!();
}

impl<'a, T> FusedIterator for CellsMut<'a, T> {}



pub struct IntoCells<T> {
  inner: Option<(XY<isize>, BaseIntoCells<Option<T>>)>
}

impl<T> Iterator for IntoCells<T> {
  type Item = (IVec2, T);

  impl_iterator_next_cells!();
}

impl<T> FusedIterator for IntoCells<T> {}



#[cfg(test)]
mod tests {
  use std::collections::HashSet;

  use super::*;

  #[test]
  fn sparsegrid_basic() {
    let mut grid = SparseGrid::new();

    grid.put(IVec2::new(0, 0), 'a');
    grid.put(IVec2::new(1, 1), 'b');
    grid.put(IVec2::new(-1, -1), 'c');

    assert_eq!(grid.width(), 3);
    assert_eq!(grid.height(), 3);
    assert_eq!(grid.min(), Some(IVec2::new(-1, -1)));
    assert_eq!(grid.max(), Some(IVec2::new(1, 1)));

    assert_eq!(grid.get(IVec2::new(0, 0)), Some(&'a'));
    assert_eq!(grid.get(IVec2::new(1, 1)), Some(&'b'));
    assert_eq!(grid.get(IVec2::new(-1, -1)), Some(&'c'));

    let elements_iter = grid.iter().cloned()
      .collect::<HashSet<char>>();
    assert!(elements_iter.contains(&'a'));
    assert!(elements_iter.contains(&'b'));
    assert!(elements_iter.contains(&'c'));

    let elements_cells = grid.cells()
      .map(|(pos, value)| (pos, *value))
      .collect::<HashSet<(IVec2, char)>>();
    assert!(elements_cells.contains(&(IVec2::new(0, 0), 'a')));
    assert!(elements_cells.contains(&(IVec2::new(1, 1), 'b')));
    assert!(elements_cells.contains(&(IVec2::new(-1, -1), 'c')));
  }

  #[test]
  fn sparsegrid_cells_indexes() {
    let mut grid = SparseGrid::new();

    for x in -3..=3 {
      for y in -3..=3 {
        let pos = IVec2::new(x, y);
        grid.put(pos, pos);
      };
    };

    for (pos, value) in grid.into_cells() {
      assert_eq!(pos, value);
    };
  }
}
