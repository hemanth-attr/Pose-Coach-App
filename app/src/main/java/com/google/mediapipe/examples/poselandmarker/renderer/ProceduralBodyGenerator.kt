package com.google.mediapipe.examples.poselandmarker.renderer

import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import kotlin.math.cos
import kotlin.math.sin

/**
 * Generates an anatomically-proportioned procedural human body from 33 smoothed pose landmarks.
 *
 * Each body part is constructed as a filled capsule (thick stroked line converted to fill path)
 * and all parts are merged via Path.op(UNION) into a single unified silhouette.
 *
 * The result is a filled Path representing the complete human body — no skeleton, no segments,
 * no rectangles — just a smooth, organic, gap-free silhouette.
 *
 * Key design decisions:
 * - All widths are proportional to measured shoulder width (never fixed pixel values)
 * - Head is a perfect circle auto-sized from ear-to-ear distance
 * - Torso is a filled quad with capsule edges for smooth boundaries
 * - Limbs use tapered capsules (thicker at proximal end)
 * - Hands and feet are rendered as oriented rounded endpoints
 */
object ProceduralBodyGenerator {

    // Reusable Paint for capsule generation — avoids per-frame allocation
    private val capsulePaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeJoin = Paint.Join.ROUND
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    /**
     * Create a capsule-shaped filled Path between two points with given thickness.
     *
     * Technique: Draw a thick stroked line, then use Paint.getFillPath() to convert
     * the stroke outline into a filled Path. This produces perfect rounded caps automatically.
     *
     * @param p1        Start point
     * @param p2        End point
     * @param thickness Diameter of the capsule
     * @return Filled Path representing the capsule
     */
    private fun capsule(p1: PointF, p2: PointF, thickness: Float): Path {
        val stroke = Path().apply {
            moveTo(p1.x, p1.y)
            lineTo(p2.x, p2.y)
        }
        val filled = Path()
        capsulePaint.strokeWidth = thickness
        capsulePaint.getFillPath(stroke, filled)
        return filled
    }

    /**
     * Create a tapered capsule — thicker at the start, thinner at the end.
     * Implemented as two overlapping capsules blended together.
     *
     * @param p1             Start point (thicker end)
     * @param p2             End point (thinner end)
     * @param thicknessStart Diameter at start
     * @param thicknessEnd   Diameter at end
     * @return Filled Path representing the tapered capsule
     */
    private fun taperedCapsule(p1: PointF, p2: PointF, thicknessStart: Float, thicknessEnd: Float): Path {
        val result = Path()
        // Create the main capsule at the average thickness
        val avgThickness = (thicknessStart + thicknessEnd) / 2f
        result.op(capsule(p1, p2, avgThickness), Path.Op.UNION)

        // Add bulge at the thicker end
        val midPoint = Geometry.lerp(p1, p2, 0.3f)
        result.op(capsule(p1, midPoint, thicknessStart), Path.Op.UNION)

        // Add taper at the thinner end
        val farPoint = Geometry.lerp(p1, p2, 0.7f)
        result.op(capsule(farPoint, p2, thicknessEnd), Path.Op.UNION)

        return result
    }

    /**
     * Generate the complete body silhouette from smoothed landmarks.
     *
     * @param pts Array of 33 smoothed landmark positions in view coordinates
     * @param incorrectLimbs Set of limbs that are incorrectly positioned
     * @return BodyRenderModel containing the main silhouette and incorrect limb paths
     */
    fun generate(pts: Array<PointF>, incorrectLimbs: Set<Limb> = emptySet()): BodyRenderModel {
        val body = Path()
        val incorrectPaths = mutableListOf<Path>()

        // ── Measure base proportions ──
        val shoulderWidth = Geometry.distance(pts[LandmarkIndex.LEFT_SHOULDER], pts[LandmarkIndex.RIGHT_SHOULDER])
            .coerceAtLeast(AnatomyRatios.MIN_SHOULDER_WIDTH_PX)
        val hipWidth = Geometry.distance(pts[LandmarkIndex.LEFT_HIP], pts[LandmarkIndex.RIGHT_HIP])
            .coerceAtLeast(AnatomyRatios.MIN_SHOULDER_WIDTH_PX * 0.6f)

        // Dynamic body rotation detection:
        // When viewed from the side, shoulder/hip width shrinks, so we use the larger of the two
        // to maintain proportional thickness
        val baseWidth = maxOf(shoulderWidth, hipWidth)

        // ══════════════════════════════════════════
        // HEAD — Perfect circle, auto-sized
        // ══════════════════════════════════════════
        val earDist = Geometry.distance(pts[LandmarkIndex.LEFT_EAR], pts[LandmarkIndex.RIGHT_EAR])
        var headRadius = earDist * AnatomyRatios.HEAD_RADIUS_FROM_EARS
        if (headRadius < baseWidth * AnatomyRatios.HEAD_MIN_RATIO) {
            headRadius = baseWidth * AnatomyRatios.HEAD_RADIUS_FALLBACK // Side-profile fallback
        }
        val headCenter = PointF(
            pts[LandmarkIndex.NOSE].x,
            pts[LandmarkIndex.NOSE].y - headRadius * 0.25f // Shift up slightly for natural look
        )
        val headPath = Path().apply {
            addCircle(headCenter.x, headCenter.y, headRadius, Path.Direction.CW)
        }
        body.op(headPath, Path.Op.UNION)

        // ══════════════════════════════════════════
        // NECK — Short rounded connector
        // ══════════════════════════════════════════
        val neckThickness = baseWidth * AnatomyRatios.NECK_WIDTH
        val midShoulder = Geometry.midpoint(pts[LandmarkIndex.LEFT_SHOULDER], pts[LandmarkIndex.RIGHT_SHOULDER])
        // Neck connects from the bottom of the head to the shoulder midpoint
        val neckTop = PointF(headCenter.x, headCenter.y + headRadius * 0.6f)
        body.op(capsule(neckTop, midShoulder, neckThickness), Path.Op.UNION)

        // ══════════════════════════════════════════
        // TORSO — Smooth capsule with dynamic width
        // ══════════════════════════════════════════
        val torsoThickness = baseWidth * AnatomyRatios.TORSO_WIDTH

        // Four edges of the torso, creating a filled hull
        val lShoulder = pts[LandmarkIndex.LEFT_SHOULDER]
        val rShoulder = pts[LandmarkIndex.RIGHT_SHOULDER]
        val lHip = pts[LandmarkIndex.LEFT_HIP]
        val rHip = pts[LandmarkIndex.RIGHT_HIP]

        // Side edges (shoulder → hip)
        body.op(capsule(lShoulder, lHip, torsoThickness), Path.Op.UNION)
        body.op(capsule(rShoulder, rHip, torsoThickness), Path.Op.UNION)

        // Top edge (shoulder → shoulder)
        body.op(capsule(lShoulder, rShoulder, torsoThickness * 0.85f), Path.Op.UNION)

        // Bottom edge (hip → hip)
        body.op(capsule(lHip, rHip, torsoThickness * 0.75f), Path.Op.UNION)

        // Fill the torso interior to eliminate any internal gaps
        val torsoFill = Path().apply {
            moveTo(lShoulder.x, lShoulder.y)
            lineTo(rShoulder.x, rShoulder.y)
            lineTo(rHip.x, rHip.y)
            lineTo(lHip.x, lHip.y)
            close()
        }
        body.op(torsoFill, Path.Op.UNION)

        // ══════════════════════════════════════════
        // LEFT ARM — Tapered capsules with hand
        // ══════════════════════════════════════════
        val upperArmWidth = baseWidth * AnatomyRatios.UPPER_ARM_WIDTH
        val forearmWidth = baseWidth * AnatomyRatios.FOREARM_WIDTH
        val handRadius = baseWidth * AnatomyRatios.HAND_RADIUS

        val leftArmPath = Path()
        // Left upper arm (shoulder → elbow) — tapered
        leftArmPath.op(
            taperedCapsule(pts[LandmarkIndex.LEFT_SHOULDER], pts[LandmarkIndex.LEFT_ELBOW], upperArmWidth, forearmWidth),
            Path.Op.UNION
        )
        // Left forearm (elbow → wrist) — tapered
        leftArmPath.op(
            taperedCapsule(pts[LandmarkIndex.LEFT_ELBOW], pts[LandmarkIndex.LEFT_WRIST], forearmWidth, forearmWidth * 0.85f),
            Path.Op.UNION
        )
        // Left hand — small circle at wrist, biased toward fingertips
        val leftHandCenter = Geometry.lerp(pts[LandmarkIndex.LEFT_WRIST], pts[LandmarkIndex.LEFT_INDEX], 0.4f)
        val leftHandPath = Path().apply {
            addCircle(leftHandCenter.x, leftHandCenter.y, handRadius, Path.Direction.CW)
        }
        leftArmPath.op(leftHandPath, Path.Op.UNION)
        // Connect hand to wrist
        leftArmPath.op(capsule(pts[LandmarkIndex.LEFT_WRIST], leftHandCenter, forearmWidth * 0.7f), Path.Op.UNION)
        
        body.op(leftArmPath, Path.Op.UNION)
        if (incorrectLimbs.contains(Limb.LEFT_ARM)) incorrectPaths.add(leftArmPath)

        // ══════════════════════════════════════════
        // RIGHT ARM — Tapered capsules with hand
        // ══════════════════════════════════════════
        val rightArmPath = Path()
        rightArmPath.op(
            taperedCapsule(pts[LandmarkIndex.RIGHT_SHOULDER], pts[LandmarkIndex.RIGHT_ELBOW], upperArmWidth, forearmWidth),
            Path.Op.UNION
        )
        rightArmPath.op(
            taperedCapsule(pts[LandmarkIndex.RIGHT_ELBOW], pts[LandmarkIndex.RIGHT_WRIST], forearmWidth, forearmWidth * 0.85f),
            Path.Op.UNION
        )
        val rightHandCenter = Geometry.lerp(pts[LandmarkIndex.RIGHT_WRIST], pts[LandmarkIndex.RIGHT_INDEX], 0.4f)
        val rightHandPath = Path().apply {
            addCircle(rightHandCenter.x, rightHandCenter.y, handRadius, Path.Direction.CW)
        }
        rightArmPath.op(rightHandPath, Path.Op.UNION)
        rightArmPath.op(capsule(pts[LandmarkIndex.RIGHT_WRIST], rightHandCenter, forearmWidth * 0.7f), Path.Op.UNION)

        body.op(rightArmPath, Path.Op.UNION)
        if (incorrectLimbs.contains(Limb.RIGHT_ARM)) incorrectPaths.add(rightArmPath)

        // ══════════════════════════════════════════
        // LEFT LEG — Tapered capsules with foot
        // ══════════════════════════════════════════
        val thighWidth = baseWidth * AnatomyRatios.THIGH_WIDTH
        val calfWidth = baseWidth * AnatomyRatios.CALF_WIDTH
        val footWidth = baseWidth * AnatomyRatios.FOOT_WIDTH

        val leftLegPath = Path()
        // Left thigh (hip → knee)
        leftLegPath.op(
            taperedCapsule(pts[LandmarkIndex.LEFT_HIP], pts[LandmarkIndex.LEFT_KNEE], thighWidth, calfWidth),
            Path.Op.UNION
        )
        // Left calf (knee → ankle)
        leftLegPath.op(
            taperedCapsule(pts[LandmarkIndex.LEFT_KNEE], pts[LandmarkIndex.LEFT_ANKLE], calfWidth, calfWidth * 0.8f),
            Path.Op.UNION
        )
        // Left foot — oriented capsule from heel to toe
        leftLegPath.op(
            capsule(pts[LandmarkIndex.LEFT_HEEL], pts[LandmarkIndex.LEFT_FOOT_INDEX], footWidth),
            Path.Op.UNION
        )
        // Connect ankle to foot
        leftLegPath.op(
            capsule(pts[LandmarkIndex.LEFT_ANKLE], pts[LandmarkIndex.LEFT_HEEL], calfWidth * 0.7f),
            Path.Op.UNION
        )

        body.op(leftLegPath, Path.Op.UNION)
        if (incorrectLimbs.contains(Limb.LEFT_LEG)) incorrectPaths.add(leftLegPath)

        // ══════════════════════════════════════════
        // RIGHT LEG — Tapered capsules with foot
        // ══════════════════════════════════════════
        val rightLegPath = Path()
        rightLegPath.op(
            taperedCapsule(pts[LandmarkIndex.RIGHT_HIP], pts[LandmarkIndex.RIGHT_KNEE], thighWidth, calfWidth),
            Path.Op.UNION
        )
        rightLegPath.op(
            taperedCapsule(pts[LandmarkIndex.RIGHT_KNEE], pts[LandmarkIndex.RIGHT_ANKLE], calfWidth, calfWidth * 0.8f),
            Path.Op.UNION
        )
        rightLegPath.op(
            capsule(pts[LandmarkIndex.RIGHT_HEEL], pts[LandmarkIndex.RIGHT_FOOT_INDEX], footWidth),
            Path.Op.UNION
        )
        rightLegPath.op(
            capsule(pts[LandmarkIndex.RIGHT_ANKLE], pts[LandmarkIndex.RIGHT_HEEL], calfWidth * 0.7f),
            Path.Op.UNION
        )

        body.op(rightLegPath, Path.Op.UNION)
        if (incorrectLimbs.contains(Limb.RIGHT_LEG)) incorrectPaths.add(rightLegPath)

        return BodyRenderModel(body, incorrectPaths)
    }
}
