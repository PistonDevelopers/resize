use resize::Pixel::Gray8;
use resize::Type::Triangle;
use resize::Type::Lanczos3;
use std::fs;
use std::path::Path;

fn load_png(mut data: &[u8]) -> (usize, usize, Vec<u8>) {
    let decoder = png::Decoder::new(&mut data);
    let (info, mut reader) = decoder.read_info().unwrap();
    let mut src = vec![0; info.buffer_size()];
    reader.next_frame(&mut src).unwrap();

    (info.width as usize, info.height as usize, src)
}

fn write_png(path: &Path, w2: usize, h2: usize, pixels: &[u8]) {
    let outfh = fs::File::create(path).unwrap();
    let encoder = png::Encoder::new(outfh, w2 as u32, h2 as u32);
    encoder.write_header().unwrap().write_image_data(pixels).unwrap();
}

fn img_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum = a.iter().cloned().zip(b.iter().cloned()).map(|(a,b)| {
        (a as i32 - b as i32).pow(2) as u32
    }).sum::<u32>();
    sum as f64 / a.len() as f64
}

fn assert_equals(img: &[u8], w2: usize, h2: usize, expected_filename: &str) {
    let (_, _, expected) = load_png(&fs::read(&expected_filename).expect(expected_filename));

    let diff = img_diff(&img, &expected);
    if diff > 0.0004 {
        let bad_file = Path::new(expected_filename).with_extension("failed-test.png");
        write_png(&bad_file, w2, h2, img);
        panic!("Test failed: {} differs by {}; see {} ", expected_filename, diff, bad_file.display());
    }
}

fn test_width(w2: usize) {
    let tiger = &include_bytes!("../examples/tiger.png")[..];
    let (w1, h1, src) = load_png(&tiger);
    let mut res1 = vec![];
    let mut res2 = vec![];

    for h2 in [1, 2, 9, 99, 999, 1555].iter().copied() {
        res1.clear();
        res1.resize(w2 * h2, 0);

        resize::new(w1, h1, w2, h2, Gray8, Lanczos3).resize(&src, &mut res1);
        assert_equals(&res1, w2, h2, &format!("tests/t{}x{}.png", w2, h2));

        res2.clear();
        res2.resize(100 * 100, 255);

        resize::new(w2, h2, 100, 100, Gray8, Triangle).resize(&res1, &mut res2);
        assert_equals(&res2, 100, 100, &format!("tests/t{}x{}-100.png", w2, h2));
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
