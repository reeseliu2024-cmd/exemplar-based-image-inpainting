import cv2
import os
from inpainting import CriminisiInpainter

IMAGE_DIR = "images"
MASK_DIR = "masks"
RESULT_DIR = "results"
INPAINTING_DIR = "inpatings"

PATCH_SIZE = 9
SNAPSHOT_INTERVAL = 20

# True: process all images
# False: process only TEST_IMAGE
RUN_ALL = True
TEST_IMAGE = "circle.png"

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(INPAINTING_DIR, exist_ok=True)



image_files = sorted([
    filename
    for filename in os.listdir(IMAGE_DIR)
    if filename.lower().endswith(".png")
])

if not RUN_ALL:
    image_files = [
        filename for filename in image_files
        if filename == TEST_IMAGE
    ]

print(f"Found {len(image_files)} images:")

for filename in image_files:
    print(f"  - {filename}")



for image_filename in image_files:

    image_name = os.path.splitext(image_filename)[0]
    image_path = os.path.join(IMAGE_DIR, image_filename)
    mask_path = os.path.join(MASK_DIR, f"{image_name}_mask.png")

    print("\n" + "=" * 60)
    print(f"Processing: {image_name}")
    print("=" * 60)

    # Step 1: Load image and mask-
    if not os.path.exists(mask_path):
        print(f"Warning: mask not found: {mask_path}")
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: cannot load {image_path}")
        continue

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Warning: cannot load {mask_path}")
        continue

    # Check image and mask dimensions
    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: size mismatch for {image_name}")
        continue

    # Convert mask to binary
    # White (255) = target region
    # Black (0) = known region
    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    # Step 2: Create result folder
    image_result_dir = os.path.join(RESULT_DIR, image_name)
    snapshot_dir = os.path.join(image_result_dir, "intermediate")

    os.makedirs(image_result_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    # Step 3: Save original image
    cv2.imwrite(os.path.join(image_result_dir, "original.png"),image)

    # Step 4: Create and save masked image
    masked_image = image.copy()
    masked_image[mask > 0] = 255

    cv2.imwrite(os.path.join(image_result_dir,"masked.png"),masked_image)


    # Step 5: Run Criminisi inpainting
    print("Starting inpainting...")

    inpainter = CriminisiInpainter(
        masked_image.copy(),
        mask.copy(),
        patch_size=PATCH_SIZE
    )

    result, snapshots = inpainter.inpaint(
        snapshot_interval=SNAPSHOT_INTERVAL,
        verbose=True
    )


    # Step 6: Save final result in results/
    cv2.imwrite(os.path.join(image_result_dir, "inpainted.png"),result)

    # Step 7: Save final image in inpatings/
    final_image_path = os.path.join(INPAINTING_DIR, f"{image_name}.png")

    cv2.imwrite(final_image_path,result)

    # Step 8: Save intermediate snapshots
    for i, snapshot in enumerate(snapshots):

        iteration = (i + 1) * SNAPSHOT_INTERVAL
        snapshot_path = os.path.join(snapshot_dir,f"iteration_{iteration:04d}.png")
        cv2.imwrite(snapshot_path,snapshot)

    # Step 9: Print experiment summary
    print(f"Finished: {image_name}")
    print(f"Total iterations: {inpainter.iteration}")
    print(f"Number of snapshots: {len(snapshots)}")
    print(f"Final image saved to: {final_image_path}")


print("\n" + "=" * 60)
print("All selected images finished!")
print("=" * 60)
