use std::io::BufReader;
use png::BitDepth;
use png::ColorType;
use resize::Pixel;
use resize::Type::Triangle;
use rgb::FromSlice;
use std::env;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        println!("Usage: {} in.png WxH out.png", args[0]);
        return Ok(());
    }

    let decoder = png::Decoder::new(BufReader::new(File::open(&args[1])?));
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let color_type = info.color_type;
    let bit_depth = info.bit_depth;
    let (w1, h1) = (info.width as usize, info.height as usize);
    let mut src = vec![0; reader.output_buffer_size()];
    reader.next_frame(&mut src)?;

    let mut dst_dim = args[2].split('x').map(|s| s.parse());
    let (w2, h2) = (dst_dim.next().ok_or("x")??, dst_dim.next().ok_or("size is widthxheight")??);
    let mut dst = vec![0u8; w2 * h2 * color_type.samples()];

    assert_eq!(BitDepth::Eight, bit_depth);
    match color_type {
        ColorType::Grayscale => resize::new(w1, h1, w2, h2, Pixel::Gray8, Triangle)?.resize(src.as_gray(), dst.as_gray_mut())?,
        ColorType::Rgb => resize::new(w1, h1, w2, h2, Pixel::RGB8, Triangle)?.resize(src.as_rgb(), dst.as_rgb_mut())?,
        ColorType::Indexed => unimplemented!(),
        ColorType::GrayscaleAlpha => unimplemented!(),
        ColorType::Rgba => resize::new(w1, h1, w2, h2, Pixel::RGBA8, Triangle)?.resize(src.as_rgba(), dst.as_rgba_mut())?,
    };

    let outfh = File::create(&args[3])?;
    let mut encoder = png::Encoder::new(outfh, w2 as u32, h2 as u32);
    encoder.set_color(color_type);
    encoder.set_depth(bit_depth);
    encoder.write_header()?.write_image_data(&dst)?;
    Ok(())
}
