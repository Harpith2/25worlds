"use client";

import { MultiFrameFusion } from "./MultiFrameFusion";
import type { ReconstructionJob, SpatialResult } from "@/lib/types";

type Props = {
  job: ReconstructionJob | null;
  ply: ArrayBuffer | null;
  accessCode: string;
  onResult?: (result: SpatialResult) => void;
  floorEstimate?: SpatialResult | null;
};

export function SpatialAnalysis({ job, ply, accessCode, onResult, floorEstimate }: Props) {
  return (
    <section className="spatial-section" id="spatial" aria-labelledby="spatial-heading">
      <div className="section-title-row spatial-title-row">
        <h2 className="section-label" id="spatial-heading">02 / Spatial intelligence</h2>
        <p>Ground objects across viewpoints in one shared 3D world.</p>
      </div>
      <MultiFrameFusion job={job} ply={ply} accessCode={accessCode} onResult={onResult} floorEstimate={floorEstimate} />
      <p className="spatial-disclaimer">
        Fusion combines observed surface samples in Pi3X coordinates. Hidden surfaces,
        metric scale, floor orientation, and navigability are not independently verified.
      </p>
    </section>
  );
}
