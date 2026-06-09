from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def _to_uint8_stack(polar_stack: np.ndarray) -> np.ndarray:
    f = np.asarray(polar_stack, dtype=np.float32)
    lo = float(np.nanpercentile(f, 1.0))
    hi = float(np.nanpercentile(f, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((f - lo) / (hi - lo), 0.0, 1.0)
    # DICOM stores frames as rows x cols; stack input is n x theta x r.
    # We write rows=radial bins, cols=angle bins for an unwrapped polar image.
    out = (norm * 255.0).astype(np.uint8).transpose(0, 2, 1)
    return out


def write_multiframe_dicom(
    polar_stack: np.ndarray,
    t_far_mm: float,
    fps: int,
    out_path: str | Path,
    dicom_cfg: dict[str, Any],
) -> dict[str, Any]:
    arr = _to_uint8_stack(polar_stack)
    n_frames, rows, cols = arr.shape

    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    now = datetime.now(timezone.utc)
    ds = FileDataset(str(out_path), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SpecificCharacterSet = "ISO_IR 100"
    ds.ImageType = r"DERIVED\PRIMARY\IVUS\SIMULATED"
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = generate_uid()

    ds.PatientName = dicom_cfg.get("patient_name", "PAT23^SIMULATED")
    ds.PatientID = dicom_cfg.get("patient_id", "PAT23_SIM")
    ds.Modality = "US"
    ds.Manufacturer = dicom_cfg.get("manufacturer", "I4H")
    ds.ManufacturerModelName = dicom_cfg.get("model_name", "CTA_TO_IVUS_SIM")
    ds.StudyDescription = dicom_cfg.get("study_description", "Simulated IVUS Pullback From CTA")
    ds.SeriesDescription = dicom_cfg.get("series_description", "PAT23 Aorta_Extended IVUS Pullback")
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.SeriesDate = ds.StudyDate
    ds.ContentDate = ds.StudyDate
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesTime = ds.StudyTime
    ds.ContentTime = ds.StudyTime
    ds.AcquisitionDateTime = now.strftime("%Y%m%d%H%M%S")

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.NumberOfFrames = str(int(n_frames))
    ds.FrameIncrementPointer = (0x0018, 0x1063)  # FrameTime
    ds.FrameTime = float(1000.0 / max(int(fps), 1))
    ds.CineRate = int(fps)
    ds.ImageComments = "Synthetic IVUS pullback generated from CTA lumen segmentation"
    ds.DepthOfScanField = float(t_far_mm)

    # Ultrasound region sequence (minimal, display-centric metadata).
    region = Dataset()
    region.RegionSpatialFormat = 1
    region.RegionDataType = 1
    region.RegionFlags = 0
    region.RegionLocationMinX0 = 0
    region.RegionLocationMinY0 = 0
    region.RegionLocationMaxX1 = int(cols - 1)
    region.RegionLocationMaxY1 = int(rows - 1)
    region.ReferencePixelX0 = 0
    region.ReferencePixelY0 = 0
    region.PhysicalUnitsXDirection = 3  # cm
    region.PhysicalUnitsYDirection = 3  # cm
    region.PhysicalDeltaX = float((360.0 / max(cols, 1)) / 10.0)  # display-scale only
    region.PhysicalDeltaY = float((t_far_mm / max(rows, 1)) / 10.0)
    ds.SequenceOfUltrasoundRegions = [region]

    # Volcano-style private creator and key tags used by existing metadata tooling.
    ds.add_new((0x0029, 0x0010), "LO", "VOLCANO-PCDE 1.0")
    ds.add_new((0x0029, 0x1001), "FD", float(dicom_cfg.get("gain_slider_ref", 54)))
    ds.add_new((0x0029, 0x1003), "FD", float(2.0 * t_far_mm))
    ds.add_new((0x0029, 0x1007), "US", 0)  # AR-OFF equivalent for synthetic output

    ds.PixelData = arr.tobytes()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pydicom.dcmwrite(str(out), ds, write_like_original=False)
    return {
        "path": str(out),
        "n_frames": int(n_frames),
        "rows": int(rows),
        "cols": int(cols),
        "fps": int(fps),
        "t_far_mm": float(t_far_mm),
    }
