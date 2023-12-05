use resize::Pixel::*;
use resize::Type::*;
use std::fs;
use std::path::Path;

fn load_png(mut data: &[u8]) -> Result<(usize, usize, Vec<u8>), png::DecodingError> {
    let decoder = png::Decoder::new(&mut data);
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    let w = info.width as usize;
    let h = info.height as usize;
    let mut src = vec![0; reader.output_buffer_size()];
    reader.next_frame(&mut src)?;

    Ok((w, h, src))
}

fn write_png(path: &Path, w2: usize, h2: usize, pixels: &[u8]) {
    let outfh = fs::File::create(path).unwrap();
    let encoder = png::Encoder::new(outfh, w2 as u32, h2 as u32);
    encoder.write_header().unwrap().write_image_data(pixels).unwrap();
}

fn img_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum = a.iter().copied().zip(b.iter().copied()).map(|(a,b)| {
        (i32::from(a) - i32::from(b)).pow(2) as u32
    }).sum::<u32>();
    f64::from(sum) / a.len() as f64
}

fn assert_equals(img: &[u8], w2: usize, h2: usize, expected_filename: &str) {
    assert_eq!(img.len(), w2 * h2);
    assert!(w2 > 0 && h2 > 0);
    let (_, _, expected) = load_png(&fs::read(expected_filename).expect(expected_filename)).expect(expected_filename);

    let diff = img_diff(img, &expected);
    if diff > 0.0004 {
        let bad_file = Path::new(expected_filename).with_extension("failed-test.png");
        write_png(&bad_file, w2, h2, img);
        panic!("Test failed: {} differs by {}; see {} ", expected_filename, diff, bad_file.display());
    }
}

#[test]
fn test_filter_init() {
    let _ = resize::new(3, 3, 25, 25, RGBF64, Point);
    let _ = resize::new(10, 10, 5, 5, RGBF64, Point);
    let _ = resize::new(3, 3, 25, 25, RGBF32, Triangle);
    let _ = resize::new(10, 10, 5, 5, RGBF32, Triangle);
    let _ = resize::new(3, 3, 25, 25, RGBAF64, Catrom);
    let _ = resize::new(10, 10, 5, 5, RGBAF64, Catrom);
    let _ = resize::new(3, 3, 25, 25, RGBAF32, Mitchell);
    let _ = resize::new(10, 10, 5, 5, RGBAF32, Mitchell);
    let _ = resize::new(3, 4, 25, 99, GrayF64, Lanczos3);
    let _ = resize::new(99, 70, 5, 1, GrayF64, Lanczos3);
}

fn test_width(w2: usize) {
    use rgb::FromSlice;

    let tiger = &include_bytes!("../examples/tiger.png")[..];
    let (w1, h1, src) = load_png(tiger).unwrap();
    let mut res1 = vec![];
    let mut res2 = vec![];
    let mut res3 = vec![0; 80*120];

    for h2 in [1, 2, 9, 99, 999, 1555].iter().copied() {
        res1.clear();
        res1.resize(w2 * h2, 0);

        resize::new(w1, h1, w2, h2, Gray8, Lanczos3).unwrap().resize(src.as_gray(), res1.as_gray_mut()).unwrap();
        assert_equals(&res1, w2, h2, &format!("tests/t{w2}x{h2}.png"));

        res2.clear();
        res2.resize(100 * 100, 255);

        resize::new(w2, h2, 100, 100, Gray8, Triangle).unwrap().resize(res1.as_gray(), res2.as_gray_mut()).unwrap();
        assert_equals(&res2, 100, 100, &format!("tests/t{w2}x{h2}-100.png"));

        resize::new(100, 100, 80, 120, Gray8, Point).unwrap().resize(res2.as_gray(), res3.as_gray_mut()).unwrap();
        assert_equals(&res3, 80, 120, &format!("tests/t{w2}x{h2}-point.png"));
    }
}

#[test]
fn test_w2000() {
    test_width(2000);
}
#[test]
fn test_w1000() {
    test_width(1000);
}
#[test]
fn test_w100() {
    test_width(100);
}
#[test]
fn test_w10() {
    test_width(10);
}
#[test]
fn test_w2() {
    test_width(2);
}
#[test]
fn test_w1() {
    test_width(1);
}

#[test]
fn can_name_type() {
    let _: resize::Resizer<resize::formats::Gray<u8, u8>> = resize::new(10, 10, 100, 100, Gray8, Triangle).unwrap();
}
