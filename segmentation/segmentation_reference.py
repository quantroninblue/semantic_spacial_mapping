import cv2
import numpy as np
import time

from ultralytics import YOLO

from geometry.obb.obb_reference import (
    OBBEstimator
)


class SegmentationModule:

    def __init__(

        self,

        model_path="yolov8n-seg.pt",

        confidence_threshold=0.45,

        minimum_mask_area=1500
    ):

        print(
            "\nLoading YOLOv8 segmentation model..."
        )

        self.model = YOLO(
            model_path
        )

        self.confidence_threshold = (
            confidence_threshold
        )

        self.minimum_mask_area = (
            minimum_mask_area
        )

        self.obb_estimator = OBBEstimator(

            fx=500.87,

            fy=501.20
        )

        print(
            "Segmentation model loaded."
        )

    # ========================================================
    # Largest connected contour
    # ========================================================

    def _extract_largest_contour(

        self,

        mask_u8
    ):

        contours, _ = cv2.findContours(

            mask_u8,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:

            return None

        largest = max(

            contours,

            key=cv2.contourArea
        )

        return largest

    # ========================================================
    # Connected component cleanup
    # ========================================================

    def _keep_largest_component(

        self,

        mask_u8
    ):

        num_labels, labels, stats, _ = (

            cv2.connectedComponentsWithStats(
                mask_u8,
                connectivity=8
            )
        )

        if num_labels <= 1:

            return mask_u8

        largest_idx = 1

        largest_area = 0

        for idx in range(1, num_labels):

            area = stats[
                idx,
                cv2.CC_STAT_AREA
            ]

            if area > largest_area:

                largest_area = area

                largest_idx = idx

        cleaned = np.zeros_like(
            mask_u8
        )

        cleaned[
            labels == largest_idx
        ] = 255

        return cleaned

    # ========================================================
    # OBB estimation
    # ========================================================

    def _estimate_instance_geometry(

        self,

        mask_u8
    ):

        obb = self.obb_estimator.estimate_obb(

            binary_mask=mask_u8,

            depth_m=1.0
        )

        return obb

    # ========================================================
    # OBB overlay
    # ========================================================

    def _draw_obb(

        self,

        frame,

        obb
    ):

        if obb is None:
            return frame

        corners = obb[
            "box_points"
        ]

        cv2.polylines(

            frame,

            [corners],

            True,

            (0, 255, 255),

            2
        )

        center = (

            int(obb["center_x"]),

            int(obb["center_y"])
        )

        cv2.circle(

            frame,

            center,

            4,

            (0, 0, 255),

            -1
        )

        label = (

            f"W:{obb['width_px']:.1f}px "

            f"H:{obb['height_px']:.1f}px "

            f"A:{obb['angle_deg']:.1f}"
        )

        cv2.putText(

            frame,

            label,

            (
                center[0] + 10,
                center[1]
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (255, 255, 255),

            1,

            cv2.LINE_AA
        )

        return frame

    # ========================================================
    # Main segmentation runtime
    # ========================================================

    def segment(

        self,

        rgb_frame
    ):

        h, w = rgb_frame.shape[:2]

        t0 = time.time()

        results = self.model(

            rgb_frame,

            verbose=False
        )

        elapsed_ms = (

            time.time() - t0

        ) * 1000.0

        overlay = rgb_frame.copy()

        masks_out = []

        obb_out = []

        class_ids_out = []

        labels_out = []

        confidences_out = []

        boxes_out = []

        # ----------------------------------------------------
        # No masks
        # ----------------------------------------------------

        if (

            len(results) == 0 or

            results[0].masks is None
        ):

            return {

                "overlay": overlay,

                "masks": [],

                "obbs": [],

                "class_ids": [],

                "labels": [],

                "confidences": [],

                "boxes": [],

                "elapsed_ms": elapsed_ms
            }

        # ----------------------------------------------------
        # YOLO outputs
        # ----------------------------------------------------

        masks = (
            results[0]
            .masks
            .data
            .cpu()
            .numpy()
        )

        confidences = (
            results[0]
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        class_ids = (
            results[0]
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        names = getattr(
            results[0],
            "names",
            getattr(self.model, "names", {})
        )

        # ----------------------------------------------------
        # Iterate detections
        # ----------------------------------------------------

        for idx, mask in enumerate(masks):

            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

            confidence = (
                confidences[idx]
            )

            if (
                confidence <
                self.confidence_threshold
            ):
                continue

            # ------------------------------------------------
            # Convert mask
            # ------------------------------------------------

            mask_u8 = (

                mask * 255

            ).astype(np.uint8)

            # ------------------------------------------------
            # Resize to RGB frame
            # ------------------------------------------------

            mask_u8 = cv2.resize(

                mask_u8,

                (w, h),

                interpolation=cv2.INTER_NEAREST
            )

            # ------------------------------------------------
            # Binary cleanup
            # ------------------------------------------------

            _, mask_u8 = cv2.threshold(

                mask_u8,

                127,

                255,

                cv2.THRESH_BINARY
            )

            # ------------------------------------------------
            # Keep largest component only
            # ------------------------------------------------

            mask_u8 = (
                self._keep_largest_component(
                    mask_u8
                )
            )

            # ------------------------------------------------
            # Contour extraction
            # ------------------------------------------------

            contour = (
                self._extract_largest_contour(
                    mask_u8
                )
            )

            if contour is None:
                continue

            # ------------------------------------------------
            # Area filtering
            # ------------------------------------------------

            contour_area = (
                cv2.contourArea(
                    contour
                )
            )

            if (
                contour_area <
                self.minimum_mask_area
            ):
                continue

            # ------------------------------------------------
            # Save mask
            # ------------------------------------------------

            masks_out.append(
                mask_u8
            )

            class_id = int(
                class_ids[idx]
            )

            class_ids_out.append(
                class_id
            )

            labels_out.append(
                str(
                    names.get(
                        class_id,
                        class_id
                    )
                )
                if isinstance(names, dict)
                else str(class_id)
            )

            confidences_out.append(
                float(confidence)
            )

            boxes_out.append(
                tuple(
                    int(v)
                    for v in boxes[idx]
                )
            )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            colored = np.zeros_like(
                rgb_frame
            )

            colored[:, :, 1] = mask_u8

            overlay = cv2.addWeighted(

                overlay,

                1.0,

                colored,

                0.4,

                0
            )

            # ------------------------------------------------
            # OBB estimation
            # ------------------------------------------------

            obb = (
                self._estimate_instance_geometry(
                    mask_u8
                )
            )

            if obb is None:
                continue

            obb_out.append(
                obb
            )

            overlay = self._draw_obb(

                overlay,

                obb
            )

        # ----------------------------------------------------
        # Return
        # ----------------------------------------------------

        return {

            "overlay": overlay,

            "masks": masks_out,

            "obbs": obb_out,

            "class_ids": class_ids_out,

            "labels": labels_out,

            "confidences": confidences_out,

            "boxes": boxes_out,

            "elapsed_ms": elapsed_ms
        }


# ============================================================
# Standalone Validation
# ============================================================

def main():

    module = SegmentationModule()

    frame = np.zeros(

        (640, 640, 3),

        dtype=np.uint8
    )

    cv2.rectangle(

        frame,

        (180, 180),

        (460, 460),

        (255, 255, 255),

        -1
    )

    result = module.segment(
        frame
    )

    print(
        "\n=== Segmentation + OBB Test ==="
    )

    print(

        f"\nInference Time: "

        f"{result['elapsed_ms']:.2f} ms"
    )

    print(

        f"\nMasks Found: "

        f"{len(result['masks'])}"
    )

    print(

        f"\nOBBs Estimated: "

        f"{len(result['obbs'])}"
    )

    for idx, obb in enumerate(
        result["obbs"]
    ):

        print(f"\nOBB {idx}")

        print(

            f"Center: "

            f"({obb['center_x']:.2f}, "

            f"{obb['center_y']:.2f})"
        )

        print(

            f"Width PX: "

            f"{obb['width_px']:.2f}"
        )

        print(

            f"Height PX: "

            f"{obb['height_px']:.2f}"
        )

        print(

            f"Angle: "

            f"{obb['angle_deg']:.2f}"
        )

    cv2.imshow(

        "Segmentation + OBB Overlay",

        result["overlay"]
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
