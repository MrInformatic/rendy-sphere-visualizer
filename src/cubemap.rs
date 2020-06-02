use anyhow::Error;
use image::hdr::HdrDecoder;
use rendy::core::hal::format::Swizzle;
use rendy::core::hal::image::{Kind as IKind, SamplerDesc, ViewKind};
use rendy::core::hal::Backend;
use rendy::factory::{Factory, ImageState};
use rendy::resource::CubeFace;
use rendy::texture::pixel::Rgb32Sfloat;
use rendy::texture::{MipLevels, Texture, TextureBuilder};
use std::fs::File;
use std::io::BufReader;
use std::num::NonZeroU8;
use std::path::Path;

pub struct HdrCubeMapBuilder {
    face_width: Option<u32>,
    face_height: Option<u32>,
    data: Option<Vec<Rgb32Sfloat>>,
    sampler_desc: Option<SamplerDesc>,
    mip_levels: MipLevels,
    premultiplied_alpha: bool,
    swizzle: Swizzle,
}

impl HdrCubeMapBuilder {
    pub fn new() -> Self {
        Self {
            face_width: None,
            face_height: None,
            data: None,
            sampler_desc: None,
            mip_levels: MipLevels::Levels(NonZeroU8::new(1).expect("This should never happen")),
            premultiplied_alpha: false,
            swizzle: Swizzle::NO,
        }
    }

    pub fn with_side<P: AsRef<Path>>(mut self, path: P, face: CubeFace) -> Result<Self, Error> {
        let decoder = HdrDecoder::new(BufReader::new(File::open(path)?))?;
        let metadata = decoder.metadata();

        match &self.face_width {
            None => self.face_width = Some(metadata.width),
            Some(side_width) => assert_eq!(
                side_width, &metadata.width,
                "The width of the image does not match the width of the other images"
            ),
        }

        match &self.face_height {
            None => self.face_height = Some(metadata.height),
            Some(side_height) => assert_eq!(
                side_height, &metadata.height,
                "The height of the image does not match the height of the other images"
            ),
        }

        let size = (metadata.width * metadata.height) as usize;
        if self.data.is_none() {
            self.data = Some(vec![Rgb32Sfloat::default(); size * 6]);
        }

        let data = &mut self.data.as_deref_mut().expect("This should never happen")
            [size * face as usize..size * (face as usize + 1)];
        decoder.read_image_transform(
            |pixel| Rgb32Sfloat {
                repr: pixel.to_hdr().0,
            },
            data,
        )?;

        Ok(self)
    }

    pub fn with_sampler_info(mut self, sampler_desc: SamplerDesc) -> Self {
        self.sampler_desc = Some(sampler_desc);
        self
    }

    pub fn with_mip_levels(mut self, mip_levels: MipLevels) -> Self {
        self.mip_levels = mip_levels;
        self
    }

    pub fn with_premultiplied_alpha(mut self, premultiplied_alpha: bool) -> Self {
        self.premultiplied_alpha = premultiplied_alpha;
        self
    }

    pub fn with_swizzle(mut self, swizzle: Swizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    pub fn build<B: Backend>(
        self,
        next_state: ImageState,
        factory: &mut Factory<B>,
    ) -> Result<Texture<B>, Error> {
        let data = self.data.ok_or(anyhow!("no cubemap data provided"))?;
        let width = self.face_width.ok_or(anyhow!("no cubemap data provided"))?;
        let height = self
            .face_height
            .ok_or(anyhow!("no cubemap data provided"))?
            * 6;

        let sampler_desc = self
            .sampler_desc
            .ok_or(anyhow!("no cubemap sampler info provided"))?;

        Ok(TextureBuilder::new()
            .with_data(data)
            .with_data_width(width)
            .with_data_height(height)
            .with_kind(IKind::D2(width, height, 1, 1))
            .with_mip_levels(self.mip_levels)
            .with_view_kind(ViewKind::Cube)
            .with_sampler_info(sampler_desc)
            .with_premultiplied_alpha(self.premultiplied_alpha)
            .with_swizzle(self.swizzle)
            .build(next_state, factory)
            .map_err(|e| anyhow!("{:?}", e))?)
    }
}
