#!/usr/bin/env python
"""Batch process multiple places from resolved_places.json.

緯度経度ベースでsite_idを生成し、重複を避けながら全地点を処理する。
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def generate_site_id(lat: float, lon: float, display_name: str) -> str:
    """
    Generate site_id from latitude and longitude.
    
    Format: {lat}_{lon} (rounded to 4 decimal places)
    Example: 35.6895_139.6917
    """
    lat_rounded = round(lat, 4)
    lon_rounded = round(lon, 4)
    return f"{lat_rounded}_{lon_rounded}"


def check_data_exists(data_dir: Path, site_id: str) -> bool:
    """Check if data already exists for this site_id."""
    site_path = data_dir / site_id
    if not site_path.exists():
        return False
    
    # Check if required files exist
    required_files = ["buildings_binary.npy", "roads_weighted.npy", "metadata.yaml"]
    return all((site_path / f).exists() for f in required_files)


def process_place(
    place: dict,
    data_dir: Path,
    config_path: Path,
    skip_existing: bool = True,
    skip_network: bool = False,
) -> dict:
    """
    Process a single place: acquire data and run analysis.
    
    Returns:
        Dictionary with processing status and results
    """
    lat = place["latitude"]
    lon = place["longitude"]
    display_name = place["display_name"]
    identifier = place.get("identifier", "")
    
    # Generate site_id from coordinates
    site_id = generate_site_id(lat, lon, display_name)
    
    result = {
        "identifier": identifier,
        "display_name": display_name,
        "latitude": lat,
        "longitude": lon,
        "site_id": site_id,
        "status": "pending",
        "error": None,
    }
    
    # Check if data already exists
    if skip_existing and check_data_exists(data_dir, site_id):
        result["status"] = "skipped_existing"
        result["message"] = f"Data already exists for {site_id}"
        print(f"⏭️  Skipping {display_name} ({site_id}): data already exists")
        return result
    
    # Step 1: Data acquisition
    print(f"\n{'='*60}")
    print(f"Processing: {display_name}")
    print(f"  Location: ({lat}, {lon})")
    print(f"  Site ID: {site_id}")
    print(f"{'='*60}")
    
    try:
        print(f"\n[1/2] Acquiring data...")
        cmd = [
            sys.executable,
            "scripts/run_acquisition.py",
            "--lat", str(lat),
            "--lon", str(lon),
            "--site-id", site_id,
            "--config", str(config_path),
            "--output-dir", str(data_dir),
        ]
        
        if skip_network:
            cmd.append("--skip-network")
        
        result_acq = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
        )
        
        if result_acq.returncode != 0:
            result["status"] = "acquisition_failed"
            result["error"] = result_acq.stderr[:500]  # Limit error message
            print(f"❌ Acquisition failed: {result['error']}")
            return result
        
        print(f"✅ Data acquisition complete")
        
        # Step 2: Analysis
        print(f"\n[2/2] Running analysis...")
        site_data_dir = data_dir / site_id
        
        cmd_analysis = [
            sys.executable,
            "scripts/run_analysis.py",
            "--data-dir", str(site_data_dir),
            "--config", str(config_path),
        ]
        
        result_analysis = subprocess.run(
            cmd_analysis,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=3600,  # 60 minutes timeout
        )
        
        if result_analysis.returncode != 0:
            result["status"] = "analysis_failed"
            result["error"] = result_analysis.stderr[:500]
            print(f"❌ Analysis failed: {result['error']}")
            return result
        
        print(f"✅ Analysis complete")
        
        # Find the output run directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        if outputs_dir.exists():
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if run_dirs:
                result["output_run_id"] = run_dirs[0].name
        
        result["status"] = "success"
        print(f"✅ Complete: {display_name} ({site_id})")
        
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Processing timeout"
        print(f"⏱️  Timeout: {display_name}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)[:500]
        print(f"❌ Error: {display_name}: {result['error']}")
    
    return result


def main():
    """Process all places from resolved_places.json."""
    parser = argparse.ArgumentParser(
        description="Batch process places from resolved_places.json"
    )
    parser.add_argument(
        "--places-file",
        type=Path,
        default=Path("data/resolved_places.json"),
        help="Path to resolved_places.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Config file path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for data",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip places where data already exists",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Process even if data exists (overwrite)",
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip network building",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of places to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load places
    if not args.places_file.exists():
        print(f"Error: Places file not found: {args.places_file}")
        return 1
    
    with open(args.places_file) as f:
        places = json.load(f)
    
    if args.limit:
        places = places[:args.limit]
    
    print("=" * 60)
    print("Batch Place Processing")
    print("=" * 60)
    print(f"Places file: {args.places_file}")
    print(f"Number of places: {len(places)}")
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 60)
    
    # Process each place
    results = []
    for i, place in enumerate(places, 1):
        print(f"\n[{i}/{len(places)}] Processing place...")
        result = process_place(
            place,
            args.data_dir,
            args.config,
            skip_existing=args.skip_existing,
            skip_network=args.skip_network,
        )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    status_counts = {}
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")
    
    # Save results
    results_file = args.data_dir / "batch_processing_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {results_file}")
    print("=" * 60)
    
    # Return non-zero if any failed
    failed = [r for r in results if r["status"] not in ["success", "skipped_existing"]]
    if failed:
        print(f"\n⚠️  {len(failed)} places failed or had errors")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

