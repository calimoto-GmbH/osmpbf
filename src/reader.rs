//! High level reader interface

use crate::blob::{BlobDecode, BlobReader};
use crate::elements::Element;
use crate::error::{new_blob_error, Result};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use crate::{BlobError, ByteOffset};

/// A reader for PBF files that gives access to the stored elements: nodes, ways and relations.
#[derive(Clone, Debug)]
pub struct ElementReader<R: Read + Send> {
    blob_iter: BlobReader<R>,
}

/// An offset to an osm element in the pbf.
/// The blob offset is the byte at which the relevant blob starts.
/// The element offset is the index of the element in that blob.
pub struct ElementOffset {
    pub blob_offset: ByteOffset,
    pub element_offset: usize,
}

impl<R: Read + Send> ElementReader<R> {
    /// Creates a new `ElementReader`.
    ///
    /// # Example
    /// ```
    /// use osmpbf::*;
    ///
    /// # fn foo() -> Result<()> {
    /// let f = std::fs::File::open("tests/test.osm.pbf")?;
    /// let buf_reader = std::io::BufReader::new(f);
    ///
    /// let reader = ElementReader::new(buf_reader);
    ///
    /// # Ok(())
    /// # }
    /// # foo().unwrap();
    /// ```
    pub fn new(reader: R) -> ElementReader<R> {
        ElementReader {
            blob_iter: BlobReader::new(reader),
        }
    }

    /// Decodes the PBF structure sequentially and calls the given closure on each element.
    /// Consider using `par_map_reduce` instead if you need better performance.
    ///
    /// # Errors
    /// Returns the first Error encountered while parsing the PBF structure.
    ///
    /// # Example
    /// ```
    /// use osmpbf::*;
    ///
    /// # fn foo() -> Result<()> {
    /// let reader = ElementReader::from_path("tests/test.osm.pbf")?;
    /// let mut ways = 0_u64;
    ///
    /// // Increment the counter by one for each way.
    /// reader.for_each(|element| {
    ///     if let Element::Way(_) = element {
    ///         ways += 1;
    ///     }
    /// })?;
    ///
    /// println!("Number of ways: {ways}");
    ///
    /// # Ok(())
    /// # }
    /// # foo().unwrap();
    /// ```
    pub fn for_each<F>(self, mut f: F) -> Result<()>
    where
        F: for<'a> FnMut(Element<'a>),
    {
        //TODO do something useful with header blocks
        for blob in self.blob_iter {
            match blob?.decode() {
                Ok(BlobDecode::OsmHeader(_)) | Ok(BlobDecode::Unknown(_)) => {}
                Ok(BlobDecode::OsmData(block)) => {
                    block.for_each_element(&mut f);
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Decodes the PBF structure sequentially and calls the given closure on each element
    /// including the offset of that element. The offset can be used to read the element again.
    /// # Errors
    /// Returns the first Error encountered while parsing the PBF structure.
    ///
    /// # Example
    /// TODO
    pub fn for_each_with_offset<F>(self, mut f: F) -> Result<()>
        where
            F: for<'a> FnMut(ElementOffset, Element<'a>),
    {
        for (blob_offset, blob) in self.blob_iter.indexed_iter() {
            match blob?.decode() {
                Ok(BlobDecode::OsmHeader(_)) | Ok(BlobDecode::Unknown(_)) => {}
                Ok(BlobDecode::OsmData(block)) => {
                    let mut element_offset = 0;
                    block.for_each_element(|x| {
                        f(ElementOffset { blob_offset, element_offset }, x);
                        element_offset += 1;
                    });
                }
                Err(e) => return Err(e),
            }
        }

        Ok(())
    }

    /// Parallel map/reduce. Decodes the PBF structure in parallel, calls the closure `map_op` on
    /// each element and then reduces the number of results to one item with the closure
    /// `reduce_op`. Similarly to the `init` argument in the `fold` method on iterators, the
    /// `identity` closure should produce an identity value that is inserted into `reduce_op` when
    /// necessary. The number of times that this identity value is inserted should not alter the
    /// result.
    ///
    /// # Errors
    /// Returns the first Error encountered while parsing the PBF structure.
    ///
    /// # Example
    /// ```
    /// use osmpbf::*;
    ///
    /// # fn foo() -> Result<()> {
    /// let reader = ElementReader::from_path("tests/test.osm.pbf")?;
    ///
    /// // Count the ways
    /// let ways = reader.par_map_reduce(
    ///     |element| {
    ///         match element {
    ///             Element::Way(_) => 1,
    ///             _ => 0,
    ///         }
    ///     },
    ///     || 0_u64,      // Zero is the identity value for addition
    ///     |a, b| a + b   // Sum the partial results
    /// )?;
    ///
    /// println!("Number of ways: {ways}");
    /// # Ok(())
    /// # }
    /// # foo().unwrap();
    /// ```
    pub fn par_map_reduce<MP, RD, ID, T>(self, map_op: MP, identity: ID, reduce_op: RD) -> Result<T>
    where
        MP: for<'a> Fn(Element<'a>) -> T + Sync + Send,
        RD: Fn(T, T) -> T + Sync + Send,
        ID: Fn() -> T + Sync + Send,
        T: Send,
    {
        self.blob_iter
            .par_bridge()
            .map(|blob| match blob?.decode() {
                Ok(BlobDecode::OsmHeader(_)) | Ok(BlobDecode::Unknown(_)) => Ok(identity()),
                Ok(BlobDecode::OsmData(block)) => {
                    Ok(block.elements().map(&map_op).fold(identity(), &reduce_op))
                }
                Err(e) => Err(e),
            })
            .reduce(
                || Ok(identity()),
                |a, b| match (a, b) {
                    (Ok(x), Ok(y)) => Ok(reduce_op(x, y)),
                    (x, y) => x.and(y),
                },
            )
    }
}

impl<R: Read + Seek + Send> ElementReader<R> {
    /// Reads and returns the element at the given offset.
    pub fn read_from_offset<FN, T>(&mut self, offset: &ElementOffset, callback: FN) -> Result<T>
    where
        FN: for<'a> Fn(Element<'a>) -> T + Sync + Send,
    {
        let blob = self.blob_iter.blob_from_offset(offset.blob_offset)?;
        let decoded = blob.to_primitiveblock()?;
        match decoded
            .elements()
            .skip(offset.element_offset)
            .next()
        {
            Some(element) => {
                Ok(callback(element))
            },
            None => Err(new_blob_error(BlobError::Empty))
        }
    }
}

impl ElementReader<BufReader<File>> {
    /// Tries to open the file at the given path and constructs an `ElementReader` from this.
    ///
    /// # Errors
    /// Returns the same errors that `std::fs::File::open` returns.
    ///
    /// # Example
    /// ```
    /// use osmpbf::*;
    ///
    /// # fn foo() -> Result<()> {
    /// let reader = ElementReader::from_path("tests/test.osm.pbf")?;
    /// # Ok(())
    /// # }
    /// # foo().unwrap();
    /// ```
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(ElementReader {
            blob_iter: BlobReader::from_path(path)?,
        })
    }
}
