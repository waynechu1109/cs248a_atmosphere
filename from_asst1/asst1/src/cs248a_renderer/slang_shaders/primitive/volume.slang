implementing "../primitive.slang";
import math;
import texture;

// The size of each marching step.
static float STEP_SIZE = 0.005;
// The density threshold to consider a hit.
static float DENSITY_THRESHOLD = 0.25;
// The delta used for finite difference normal estimation.
static float FINITE_DIFF_DELTA = 0.001;

public struct Volume
{
    public BoundingBox bound;
    public SharedTexture3D tex;
    public float4x4 modelMatrix;
    public float4x4 invModelMatrix;

    /**
     * Test ray-volume bounding box intersection.
     * @param ray The ray to test against the volume.
     * @param volumeTexBuf The texture buffer contains all the volumetrically represented models in the scene .
     * @return An Optional containing RayHitResult if the ray hits the volume, otherwise an empty Optional.
     */
    public Optional<RayHitResult> hit(Ray ray, SharedTexture3DBuffer<float4> volumeTexBuf)
    {
        // TODO: Student implementation starts here.

        // invModelMatrix: from world to local
        // modelMatrix: from local to world

        // transform ray to local space (volume data is in its own space)
        Ray local_ray = ray.transform(this.invModelMatrix);

        // intersect with local space bbox
        float2 tRangeBox = this.bound.hit(local_ray);
        if (tRangeBox.x > tRangeBox.y)
            // if no intersection (the first element is greater than the second element) -> no intersection
            return Optional<RayHitResult>();

        // the marching range
        // 確保 ray 只在「真的在 box 裡面」的那一段走
        float tMin = max(local_ray.tRange.x, tRangeBox.x);
        float tMax = min(local_ray.tRange.y, tRangeBox.y);

        float dirLen = length(local_ray.direction);  // the length of ray direction
        // "t" is not distance, t * |direction| is distance
        // to ensure that the step size match t to actually go forward with distance of STEP_SIZE
        float stepT = STEP_SIZE / max(dirLen, 1e-8);

        float prevDensity = 0.0;
        float t = tMin;

        while (t <= tMax)
        {
            float3 pLocal = local_ray.origin + t * local_ray.direction;    // the local point of this step
            float3 uvw = this.getUVW(pLocal);

            float density = 0.0;
            if (all(uvw >= 0.0) && all(uvw <= 1.0))
            {
                density = volumeTexBuf.trilinearSample(this.tex, uvw).w;   // this will return 0 when uvw is not in range
            }

            // detect threshold crossing (outside -> inside) -> hit!
            if (prevDensity < DENSITY_THRESHOLD && density >= DENSITY_THRESHOLD)
            {
                float3 pHitLocal = local_ray.origin + t * local_ray.direction;  // the hit point
                float3 uvwHit = this.getUVW(pHitLocal);                         // uvw of the hit point

                // computing normal
                float sxp = 0.0, sxn = 0.0, syp = 0.0, syn = 0.0, szp = 0.0, szn = 0.0;

                // x
                float3 uvw_xp = uvwHit + float3(FINITE_DIFF_DELTA, 0.0, 0.0);
                float3 uvw_xn = uvwHit - float3(FINITE_DIFF_DELTA, 0.0, 0.0);
                if (all(uvw_xp >= 0.0) && all(uvw_xp <= 1.0)) 
                {
                    sxp = volumeTexBuf.trilinearSample(this.tex, uvw_xp).w;
                }
                if (all(uvw_xn >= 0.0) && all(uvw_xn <= 1.0))
                {
                    sxn = volumeTexBuf.trilinearSample(this.tex, uvw_xn).w;
                }

                // y
                float3 uvw_yp = uvwHit + float3(0.0, FINITE_DIFF_DELTA, 0.0);
                float3 uvw_yn = uvwHit - float3(0.0, FINITE_DIFF_DELTA, 0.0);
                if (all(uvw_yp >= 0.0) && all(uvw_yp <= 1.0))
                {
                    syp = volumeTexBuf.trilinearSample(this.tex, uvw_yp).w;
                }
                if (all(uvw_yn >= 0.0) && all(uvw_yn <= 1.0))
                {
                    syn = volumeTexBuf.trilinearSample(this.tex, uvw_yn).w;
                }

                // z
                float3 uvw_zp = uvwHit + float3(0.0, 0.0, FINITE_DIFF_DELTA);
                float3 uvw_zn = uvwHit - float3(0.0, 0.0, FINITE_DIFF_DELTA);
                if (all(uvw_zp >= 0.0) && all(uvw_zp <= 1.0)) 
                {
                    szp = volumeTexBuf.trilinearSample(this.tex, uvw_zp).w;
                }
                if (all(uvw_zn >= 0.0) && all(uvw_zn <= 1.0)) 
                {
                    szn = volumeTexBuf.trilinearSample(this.tex, uvw_zn).w;
                }

                // 密度在 uvw 空間（texture 座標）中的梯度
                float3 gradUVW = float3(sxp - sxn, syp - syn, szp - szn);

                // compute grad in local position space (chain rule)
                // ∂ρ/∂p = (∂ρ/∂uvw) * (∂uvw/∂p) = gradUVW / extent  
                float3 extent = (this.bound.pMax - this.bound.pMin);

                // 密度在實際 3D 空間（local position）中的梯度
                float3 gradLocal = gradUVW / max(extent, float3(1e-8, 1e-8, 1e-8));
                float3 normal_local = -normalize(gradLocal);

                // transform normal to world: (M^-1)^T * n
                // float3 normal_world = normalize(
                //     mul(transpose(float3x3(this.invModelMatrix)), normal_local)
                // );
                float3 normal_world = normalize(
                    mul(transpose((this.invModelMatrix)), float4(normal_local, 0.0)).xyz
                );

                float3 pWorldHit = mul(this.modelMatrix, float4(pHitLocal, 1.0)).xyz;

                float3 toHit = pWorldHit - ray.origin;  // the vector from the origin to the hit point
                // 想知道 toHit 在 D 方向上「走了多少倍的 D」？
                // t = [(P-O) . D] / [D . D] （向量投影公式）
                float tWorld = dot(toHit, ray.direction) / max(dot(ray.direction, ray.direction), 1e-8); 

                RayHitResult result;
                result.t = tWorld;
                result.normal = normal_world;
                return Optional<RayHitResult>(result);
            }

            prevDensity = density;
            t += stepT;
        }

        // TODO: Student implementation ends here.

        return Optional<RayHitResult>();
    }

    /**
     * Get the uvw coordinates of a point in local space.
     * @param p The point in world space.
     * @return The uvw coordinates in [0, 1].
     */
    public float3 getUVW(float3 p)
    {
        // Compute uvw coordinates in [0, 1].
        float3 minBound = this.bound.pMin;
        float3 maxBound = this.bound.pMax;
        return (p - minBound) / (maxBound - minBound);
    }
};
