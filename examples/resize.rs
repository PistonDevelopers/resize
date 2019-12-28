use resize::Pixel::Gray8;
use resize::Type::Triangle;
use std::env;
use std::fs::File;

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
    let mut dst = vec![0; w2 * h2];
    // TODO(Kagami): Support RGB24, RGBA and custom filters.
    resize::resize(w1, h1, w2, h2, Gray8, Triangle, &src, &mut dst);

    let outfh = File::create(&args[3]).unwrap();
    let encoder = png::Encoder::new(outfh, w2 as u32, h2 as u32);
    encoder.write_header().unwrap().write_image_data(&dst).unwrap();
}
