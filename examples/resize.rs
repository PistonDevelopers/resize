use resize::Pixel;
use resize::Type::Triangle;
use std::env;
use std::fs::File;
use png::ColorType;
use png::BitDepth;
use rgb::FromSlice;

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return println!("Usage: {} in.png WxH out.png", args[0]);
    }

    let decoder = png::Decoder::new(File::open(&args[1]).unwrap());
    let (info, mut reader) = decoder.read_info().unwrap();
    let mut src = vec![0; info.buffer_size()];
    reader.next_frame(&mut src).unwrap();

    let (w1, h1) = (info.width as usize, info.height as usize);
    let dst_dims: Vec<_> = args[2].split("x").map(|s| s.parse().unwrap()).collect();
    let (w2, h2) = (dst_dims[0], dst_dims[1]);
    let mut dst = vec![0u8; w2 * h2 * info.color_type.samples()];

    assert_eq!(BitDepth::Eight, info.bit_depth);
    match info.color_type {
        ColorType::Grayscale => resize::new(w1, h1, w2, h2, Pixel::Gray8, Triangle).unwrap().resize(src.as_gray(), dst.as_gray_mut()).unwrap(),
        ColorType::RGB => resize::new(w1, h1, w2, h2, Pixel::RGB8, Triangle).unwrap().resize(src.as_rgb(), dst.as_rgb_mut()).unwrap(),
        ColorType::Indexed => unimplemented!(),
        ColorType::GrayscaleAlpha => unimplemented!(),
        ColorType::RGBA => resize::new(w1, h1, w2, h2, Pixel::RGBA8, Triangle).unwrap().resize(src.as_rgba(), dst.as_rgba_mut()).unwrap(),
    };

    let outfh = File::create(&args[3]).unwrap();
    let mut encoder = png::Encoder::new(outfh, w2 as u32, h2 as u32);
    encoder.set_color(info.color_type);
    encoder.set_depth(info.bit_depth);
    encoder.write_header().unwrap().write_image_data(&dst).unwrap();
}
