use std::{collections::{BTreeMap, HashMap}, fs::File, os::unix::fs::FileExt};

use sleigher::execution::Address;
use anyhow::Result;


pub trait MemoryRegion: core::fmt::Debug {
    fn read(&self, address: Address, data: &mut [u8]) -> Result<()>;
    fn write(&mut self, address: Address, data: &[u8]) -> Result<()>;
}

#[derive(Debug)]
pub struct HashSpace(HashMap<Address, u8>);

impl MemoryRegion for HashSpace {
    fn read(&self, address: Address, data: &mut [u8]) -> Result<()> {
        for (offset, byte) in data.iter_mut().enumerate() {
            *byte = *self.0.get(&Address(address.0 + offset as u64)).unwrap_or(&0);
        }
        Ok(())
    }

    fn write(&mut self, address: Address, data: &[u8]) -> Result<()> {
        for (offset, byte) in data.iter().enumerate() {
            self.0.insert(Address(address.0 + offset as u64), *byte);
        }
        Ok(())
    }
}

impl HashSpace {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}

impl Default for HashSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct FileRegion(pub File);

impl MemoryRegion for FileRegion {
    fn read(&self, address: Address, data: &mut [u8]) -> Result<()> {
        self.0.read_at(data, address.0)?;
        Ok(())
    }

    fn write(&mut self, address: Address, data: &[u8]) -> Result<()> {
        log::debug!("File write at {} <- {:02x?}", address, data);
        log::warn!("Ignoring write to FileRegion");
        Ok(())
    }
}

#[derive(Debug)]
pub struct MappedSpace(BTreeMap<Address, (Address, Box<dyn MemoryRegion>)>);

impl MappedSpace {
    pub fn new() -> MappedSpace {
        Self(BTreeMap::new())
    }

    pub fn add_mapping(&mut self, address: Address, length: u64, region: Box<dyn MemoryRegion>) {
        self.0.insert(address, (Address(address.0 + length), region));
    }
}

impl MemoryRegion for MappedSpace {
    fn read(&self, mut address: Address, mut data: &mut [u8]) -> Result<()> {
        for (base, (end, space)) in self.0.iter() {
            if address < *base {
                continue;
            }
            if *base <= address && address < *end {
                let offset = address.0 - base.0;
                let length = std::cmp::min(data.len(), (end.0 - address.0) as usize);
                space.read(Address(offset), &mut data[..length])?;
                address.0 += length as u64;
                data = &mut data[length..];
            }
            if data.len() == 0 {
                break;
            }
        }
        Ok(())
    }

    fn write(&mut self, mut address: Address, mut data: &[u8]) -> Result<()> {
        for (base, (end, space)) in self.0.iter_mut() {
            if address < *base {
                continue;
            }
            if *base <= address && address < *end {
                let offset = address.0 - base.0;
                let length = std::cmp::min(data.len(), (end.0 - address.0) as usize);
                space.write(Address(offset), &data[..length])?;
                address.0 += length as u64;
                data = &data[length..];
            }
            if data.len() == 0 {
                break;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct TraceSpace<T>(pub T);

impl<T: MemoryRegion> MemoryRegion for TraceSpace<T> {
    fn read(&self, address: Address, data: &mut [u8]) -> Result<()> {
        let result = self.0.read(address, data);
        log::debug!("{:#010x} -> {:02x?}", address.0, data);
        result
    }

    fn write(&mut self, address: Address, data: &[u8]) -> Result<()> {
        log::debug!("{:#010x} <- {:02x?}", address.0, data);
        self.0.write(address, data)
    }
}