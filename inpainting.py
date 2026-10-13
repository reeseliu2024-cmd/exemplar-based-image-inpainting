import cv2
from matplotlib.pyplot import gray
import numpy as np


class CriminisiInpainter:
    """
    Mask convention:
        1 = target / missing region
        0 = known / source region
    """

    def __init__(self, image, mask, patch_size=9):
        self.image = image.astype(np.float64)
        self.mask = (mask > 0).astype(np.uint8)

        self.original_mask = self.mask.copy()

        self.lab_image = cv2.cvtColor(
            np.clip(self.image, 0, 255).astype(np.uint8),
            cv2.COLOR_BGR2LAB
        ).astype(np.float64)

        self.patch_size = patch_size
        self.patch_radius = patch_size // 2
        self.height, self.width = self.mask.shape

        # C(p) = 0 inside Omega, 1 outside Omega
        self.confidence = 1.0 - self.mask.astype(np.float64)

        self.iteration = 0


    # 1. Find fill front delta Omega
    def compute_fill_front(self):
        """
        Find the boundary between the target region Omega
        and the known/source region.
        """

        kernel = np.ones((3, 3), np.uint8)

        eroded = cv2.erode(
            self.mask,
            kernel,
            iterations=1
        )

        fill_front = self.mask - eroded

        return fill_front

    #obtain patch coordinates
    def get_patch_bounds(self, y, x):

        r = self.patch_radius

        y1 = max(0, y - r)
        y2 = min(self.height, y + r + 1)

        x1 = max(0, x - r)
        x2 = min(self.width, x + r + 1)

        return y1, y2, x1, x2

    # 2. Confidence term C(p)
    def compute_confidence_term(self, y, x):

        y1, y2, x1, x2 = self.get_patch_bounds(y, x)
        confidence_patch = self.confidence[y1:y2, x1:x2]
        patch_area = confidence_patch.size
        C = np.sum(confidence_patch) / patch_area

        return C

    # 3. Compute image gradients
    def compute_gradients(self):

        gray = cv2.cvtColor(
            np.clip(self.image, 0, 255).astype(np.uint8),
            cv2.COLOR_BGR2GRAY
        ).astype(np.float64)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) / 8.0

        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) / 8.0

        # A valid gradient must use only known pixels.
        known = (self.mask == 0).astype(np.uint8)

        valid = cv2.erode(
            known,
            np.ones((3, 3), np.uint8),
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0
        ).astype(bool)

        grad_x[~valid] = 0.0
        grad_y[~valid] = 0.0

        return grad_x, grad_y, valid

    # 4. Compute normal n_p
    def compute_normals(self):

        mask_float = self.mask.astype(np.float64)
        nx = cv2.Sobel(mask_float,cv2.CV_64F,1,0,ksize=3)
        ny = cv2.Sobel(mask_float,cv2.CV_64F,0,1,ksize=3)

        magnitude = np.sqrt(nx**2 + ny**2)
        magnitude[magnitude == 0] = 1

        nx = nx / magnitude
        ny = ny / magnitude

        return nx, ny

    # 5. Data term D(p)
    def compute_data_term(self, y, x, grad_x, grad_y, valid, nx, ny):

        y1, y2, x1, x2 = self.get_patch_bounds(y, x)

        gx_patch = grad_x[y1:y2, x1:x2]
        gy_patch = grad_y[y1:y2, x1:x2]
        valid_patch = valid[y1:y2, x1:x2]

        magnitude = np.hypot(gx_patch, gy_patch)
        magnitude[~valid_patch] = -np.inf

        if not np.any(valid_patch):
            return 0.0

        iy, ix = np.unravel_index(np.argmax(magnitude), magnitude.shape)

        gx = gx_patch[iy, ix]
        gy = gy_patch[iy, ix]

        # Rotate gradient by 90 degrees.
        isophote_x = -gy
        isophote_y = gx

        dot_product = (isophote_x * nx[y, x] + isophote_y * ny[y, x])

        alpha = 255.0

        return abs(dot_product) / alpha

    # 6. Compute priorities
    def compute_priorities(self, fill_front):

        grad_x, grad_y, valid = self.compute_gradients()
        nx, ny = self.compute_normals()

        points = np.argwhere(fill_front > 0)
        priorities = []

        for y, x in points:

            C = self.compute_confidence_term(y, x)
            D = self.compute_data_term(y, x,grad_x, grad_y, valid,nx, ny)
            P = C * D

            priorities.append((P, C, D, y, x))

        return priorities

    # 7. Find best exemplar patch
    def find_best_exemplar(self, target_y, target_x):

        ty1, ty2, tx1, tx2 = self.get_patch_bounds(target_y, target_x)

        target_patch = self.image[ty1:ty2, tx1:tx2]
        target_mask = self.mask[ty1:ty2, tx1:tx2]

        known_pixels = (target_mask == 0)

        patch_h, patch_w = target_patch.shape[:2]

        best_error = np.inf
        best_patch = None

        # Search the source region Phi.
        for y1 in range(0, self.height - patch_h + 1):

            for x1 in range(0,self.width - patch_w + 1):

                y2 = y1 + patch_h
                x2 = x1 + patch_w

                source_mask = self.original_mask[y1:y2, x1:x2]

                # Valid exemplar must be completely
                # inside the known/source region.
                if np.any(source_mask != 0):
                    continue

                source_patch = self.image[y1:y2, x1:x2]

                target_patch_lab = self.lab_image[ty1:ty2, tx1:tx2]
                source_patch_lab = self.lab_image[y1:y2, x1:x2]

                difference = target_patch_lab - source_patch_lab
                diff_known = difference[known_pixels]

                error = np.sum(diff_known ** 2)

                if error < best_error:

                    best_error = error
                    best_patch = source_patch.copy()

        return best_patch

    # 8. Copy best exemplar into target patch
    def fill_patch(self, target_y,target_x,source_patch, patch_confidence):

        y1, y2, x1, x2 = self.get_patch_bounds(target_y,target_x)

        target_mask = self.mask[y1:y2, x1:x2]
        unknown_pixels = (target_mask == 1)
        target_patch = self.image[y1:y2, x1:x2]

        # Copy only unknown pixels.
        target_patch[unknown_pixels] = \
            source_patch[unknown_pixels]

        self.image[y1:y2, x1:x2] = target_patch

        self.lab_image = cv2.cvtColor(
            np.clip(self.image, 0, 255).astype(np.uint8),
            cv2.COLOR_BGR2LAB
        ).astype(np.float64)

        # Update confidence values.
        confidence_patch = self.confidence[y1:y2, x1:x2]
        confidence_patch[unknown_pixels] = \
            patch_confidence

        self.confidence[y1:y2,x1:x2] = confidence_patch

        # These pixels are now filled.
        target_mask[unknown_pixels] = 0
        self.mask[y1:y2, x1:x2] = target_mask

    # 9. Complete inpainting algorithm
    def inpaint(
        self,
        snapshot_interval=20,
        verbose=True
    ):

        snapshots = []

        while np.any(self.mask):

            self.iteration += 1

            # Step 1:compute delta Omega
            fill_front = self.compute_fill_front()

            # Step 2:calculate P(p)
            priorities = self.compute_priorities(fill_front)

            if len(priorities) == 0:
                raise RuntimeError(
                    "No valid fill-front pixels found."
                )

            # Highest-priority patch
            priorities.sort(
                key=lambda item: item[0],
                reverse=True
            )

            P, C, D, y, x = priorities[0]

            # Step 3:find best source exemplar
            best_patch = self.find_best_exemplar(y, x)

            if best_patch is None:
                raise RuntimeError(
                    "No valid source exemplar found."
                )

            # Step 4: copy exemplar + update confidence
            self.fill_patch(y, x, best_patch, C)

            if verbose:

                remaining = np.sum(self.mask)

                print(
                    f"Iteration {self.iteration:4d} | "
                    f"P={P:.5f} | "
                    f"C={C:.5f} | "
                    f"D={D:.5f} | "
                    f"Remaining={remaining}"
                )

            if (
                snapshot_interval is not None
                and
                self.iteration % snapshot_interval == 0
            ):

                snapshots.append(
                    np.clip(
                        self.image,
                        0,
                        255
                    ).astype(np.uint8).copy()
                )

        result = np.clip(
            self.image,
            0,
            255
        ).astype(np.uint8)

        return result, snapshots


