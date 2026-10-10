import cv2
import os

from inpainting import CriminisiInpainter


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "images"
MASK_DIR = "masks"
RESULT_DIR = "results"

PATCH_SIZE = 9
SNAPSHOT_INTERVAL = 20


# ============================================================
# Create output directory
# ============================================================

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Find all PNG images
# ============================================================

image_files = sorted([
    filename
    for filename in os.listdir(IMAGE_DIR)
    if filename.lower().endswith(".png")
])

print(f"Found {len(image_files)} images:")
for filename in image_files:
    print(f"  - {filename}")


# ============================================================
# Process every image
# ============================================================

for image_filename in image_files:

    # --------------------------------------------------------
    # Get image name
    # aerial.png -> aerial
    # --------------------------------------------------------

    image_name = os.path.splitext(image_filename)[0]

    image_path = os.path.join(
        IMAGE_DIR,
        image_filename
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{image_name}_mask.png"
    )

    print("\n" + "=" * 60)
    print(f"Processing: {image_name}")
    print("=" * 60)

    # --------------------------------------------------------
    # Check whether corresponding mask exists
    # --------------------------------------------------------

    if not os.path.exists(mask_path):
        print(
            f"Warning: mask not found for {image_filename}"
        )
        print(
            f"Expected: {mask_path}"
        )
        print("Skipping this image.")
        continue


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print(
            f"Warning: could not load {image_path}"
        )
        continue


    # --------------------------------------------------------
    # Load mask
    #
    # white = target / missing region
    # black = known / source region
    # --------------------------------------------------------

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print(
            f"Warning: could not load {mask_path}"
        )
        continue


    # --------------------------------------------------------
    # Make sure image and mask have the same size
    # --------------------------------------------------------

    if image.shape[:2] != mask.shape[:2]:

        print(
            f"Warning: size mismatch for {image_name}"
        )

        print(
            f"Image size: {image.shape[:2]}"
        )

        print(
            f"Mask size: {mask.shape[:2]}"
        )

        print("Skipping this image.")
        continue


    # --------------------------------------------------------
    # Convert mask to binary
    #
    # 255 = target region
    # 0   = known region
    # --------------------------------------------------------

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )


    # --------------------------------------------------------
    # Create a folder for this image
    #
    # results/aerial/
    # results/baseball/
    # ...
    # --------------------------------------------------------

    image_result_dir = os.path.join(
        RESULT_DIR,
        image_name
    )

    os.makedirs(
        image_result_dir,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save original image
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            image_result_dir,
            "original.png"
        ),
        image
    )


    # --------------------------------------------------------
    # Create and save masked image
    # --------------------------------------------------------

    masked_image = image.copy()

    masked_image[mask > 0] = 255

    cv2.imwrite(
        os.path.join(
            image_result_dir,
            "masked.png"
        ),
        masked_image
    )


    # --------------------------------------------------------
    # Run Criminisi exemplar-based inpainting
    # --------------------------------------------------------

    print("Starting inpainting...")

    inpainter = CriminisiInpainter(
        image,
        mask,
        patch_size=PATCH_SIZE
    )

    result, snapshots = inpainter.inpaint(
        snapshot_interval=SNAPSHOT_INTERVAL,
        verbose=True
    )


    # --------------------------------------------------------
    # Save final result
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            image_result_dir,
            "inpainted.png"
        ),
        result
    )


    # --------------------------------------------------------
    # Save intermediate snapshots
    # --------------------------------------------------------

    snapshot_dir = os.path.join(
        image_result_dir,
        "intermediate"
    )

    os.makedirs(
        snapshot_dir,
        exist_ok=True
    )

    for i, snapshot in enumerate(snapshots):

        cv2.imwrite(
            os.path.join(
                snapshot_dir,
                f"iteration_{(i + 1) * SNAPSHOT_INTERVAL:04d}.png"
            ),
            snapshot
        )


    print(
        f"Finished: {image_name}"
    )

    print(
        f"Number of snapshots: {len(snapshots)}"
    )


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("All images finished!")
print("=" * 60)