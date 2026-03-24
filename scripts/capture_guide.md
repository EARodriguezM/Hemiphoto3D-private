# Image Capture Guide for 3D Reconstruction

A practical field guide for capturing small biological specimens (2–10 cm) for photogrammetric 3D reconstruction using a smartphone.

---

## 1. Equipment

### Required

- **Smartphone** with ≥ 12 MP rear camera (iPhone 13+, Pixel 7+, Galaxy S22+)
- **Tripod or phone clamp** — stability is critical; handheld introduces motion blur
- **Diffused lighting** — LED light panel, ring light with diffuser, or a DIY lightbox (translucent plastic bin + desk lamp)
- **Matte background** — gray, beige, or earth-toned textured paper or fabric. Avoid pure white and glossy surfaces; they starve the feature matcher of keypoints
- **Scale bar** — printed ruler, 1 cm grid card, or calibration target visible in at least 3 images

### Recommended

- **Turntable** — manual lazy susan, motorized turntable, or DIY (two plates + marbles)
- **Modeling clay or insect pins** — to hold the specimen at the turntable center
- **Color checker card** — for post-capture white-balance correction
- **Remote shutter / timer** — eliminates hand vibration when pressing the shutter button

---

## 2. Camera Settings

Configure these **before** you start shooting. Most smartphone camera apps expose them via a "Pro" or "Manual" mode.

| Setting | Action | Why |
|---------|--------|-----|
| Focus | Tap-and-hold on the specimen to lock | Prevents focus hunting between shots |
| Exposure | Lock after metering on the specimen | Prevents brightness flickering |
| White balance | Lock to a fixed value (e.g., daylight / 5500 K) | Prevents color shifts between shots |
| Resolution | Highest quality JPEG (or RAW if supported) | Maximizes detail for feature detection |
| HDR | **Disable** | HDR compositing can cause ghosting if anything moves slightly |
| Lens | Use the **main rear camera** (1×) | Ultra-wide lenses introduce strong barrel distortion |
| Flash | **Disable** | Flash creates harsh specular highlights |
| Zoom | **Do not use digital zoom** | It just crops and upscales — move the camera closer instead |

---

## 3. Turntable Protocol (Recommended)

This method produces the most consistent results because the camera stays fixed and the specimen rotates.

### Setup

1. Place the turntable on a stable surface with the matte background underneath and behind
2. Mount the specimen at the **center** of the turntable using clay or pins
3. Position the camera on a tripod at **15–30 cm** from the specimen
4. Set up diffused lighting from two sides to minimize shadows
5. Lock focus, exposure, and white balance

### Capture Rings

Capture three elevation rings by adjusting the tripod height between rings:

| Ring | Camera Elevation | Number of Shots | Rotation Increment |
|------|-----------------|-----------------|-------------------|
| Low | ~15° above horizontal | 36 | 10° |
| Mid | ~45° above horizontal | 24 | 15° |
| High | ~75° (nearly top-down) | 12 | 30° |

**Total: 72 images**

### Tips

- Mark the turntable edge at 0° so you can count increments
- Rotate the turntable slowly and let it settle before shooting
- Include the scale bar visible in at least 3 shots (beginning, middle, end)
- If the specimen has a complex underside, flip it and repeat a partial set

---

## 4. Handheld Protocol (Alternative)

Use this method when a turntable is not available.

### Procedure

1. Place the specimen on the matte background. Do **not** move it during the shoot
2. Walk around the specimen in three rings at different heights:
   - **Low ring** — camera near table level, angled slightly up
   - **Mid ring** — camera at ~45° looking down
   - **High ring** — camera nearly overhead
3. Maintain **60–80% overlap** between adjacent shots — each surface point should be visible in at least 3 images
4. Move the **camera**, never the specimen
5. Keep a steady pace and constant distance from the specimen

### Counts

- **Minimum**: 50 images
- **Recommended**: 80–100 images
- More overlap is always better; too few images cause holes in the reconstruction

---

## 5. Common Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| White or reflective background | Feature matching fails — not enough texture | Use matte gray/beige textured paper |
| Moving the specimen between shots | Reconstruction fails entirely | Anchor the specimen; move only the camera or turntable |
| Inconsistent lighting | Color artifacts and brightness jumps | Lock exposure and white balance; use constant diffused light |
| Too few images | Holes and gaps in the mesh | Capture ≥ 50 images with 60–80% overlap |
| All photos from the same height | Top and bottom surfaces are missing | Use at least 2–3 elevation angles |
| Shiny or wet specimen | Specular reflections break multi-view stereo | Blot dry; dust with corn starch or matte powder if needed |
| Digital zoom | Loss of detail, noise amplification | Move the camera closer instead |
| HDR enabled | Ghosting artifacts from composite exposures | Disable HDR in camera settings |
| Blurry images | Features cannot be detected | Use a tripod, remote shutter, or timer |

---

## 6. Quick Validation Checklist

Before leaving the capture session, review your images on the phone:

- [ ] **Coverage**: Flip through all images — does every surface of the specimen appear in ≥ 3 shots?
- [ ] **Elevation**: Are there at least 2 different camera elevation angles (preferably 3)?
- [ ] **Scale bar**: Is the scale bar clearly visible in at least 3 images?
- [ ] **Exposure consistency**: Do the images look consistent in brightness and color?
- [ ] **Sharpness**: Zoom in on a few images — are fine details crisp, not blurry?
- [ ] **Background**: Is the background matte and textured (not white/glossy)?
- [ ] **Count**: Do you have at least 50 images (72 recommended with turntable)?

If any check fails, re-shoot the missing angles or problematic images before packing up. It is much easier to capture additional images now than to return later.

---

## Running the Preprocessing Script

After transferring images to your computer, run:

```bash
python scripts/preprocess_images.py \
    --input raw_photos/ \
    --output processed/ \
    --max-size 3200 \
    --normalize-exposure \
    --check-quality
```

The script will auto-rotate images based on EXIF data, resize them, flag blurry shots, and produce a quality report. See `python scripts/preprocess_images.py --help` for all options.
