"""Configuration for the temporal fall-detection pipeline."""

SEED = 42
TEST_SIZE = 0.2
WINDOW_SIZE = 55
WINDOW_STRIDE = 15
FOLDS = 4
BATCH_SIZE = 16
EPOCHS = 100
MIN_DELTA = 1e-4
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
THRESHOLD = 0.5
KERNEL_SIZE = 5
# Preserve the temporal length for odd kernel sizes.
PADDING = KERNEL_SIZE // 2
DROPOUT = 0.2
MINIMUM_ALLOWED_RECALL = 0.95
NUM_WORKERS=1

# Column names for the raw CSV files (no header row in the original files).
COLUMN_NAMES = [
    "sequence_name",        # Unique identifier for each video (sequence)
    "frame_number",         # Frame index within the sequence
    "label",                # Ground-truth label
    "HeightWidthRatio",     # Bounding box height to width ratio
    "MajorMinorRatio",      # Major to minor axis ratio of the fitted ellipse
    "BoundingBoxOccupancy", # Ratio of how bounding box is occupied by the silhouette
    "MaxStdXZ",             # Standard deviation of pixels from X and Z axes
    "HHmaxRatio",           # Human height in frame to human height while standing ratio
    "H",                    # Actual height (in mm)
    "D",                    # Distance of person center to the floor (in mm)
    "P40"                   # Ratio of the number of the point clouds belonging to the
                            # cuboid of 40 cm height and placed on the floor to the number
                            # of the point clouds belonging to the cuboid of height equal
                            # to person's height.
]

# Feature columns used as inputs to the temporal CNN model.
# These are a subset of the columns in COLUMN_NAMES.
FEATURES = [
    "HeightWidthRatio",
    "MajorMinorRatio",
    "BoundingBoxOccupancy",
    "MaxStdXZ",
    "HHmaxRatio",
    "H",
    "D",
    "P40"
]
