implementing "../math.slang";

public struct BoundingBox
{
    public float3 pMin;
    public float3 pMax;

    public static BoundingBox union
    (BoundingBox a, BoundingBox b)
    {
        float3 uMin = min(a.pMin, b.pMin);
        float3 uMax = max(a.pMax, b.pMax);
        return BoundingBox(uMin, uMax);
    }

    public __init(float3 pMin, float3 pMax)
    {
        this.pMin = pMin;
        this.pMax = pMax;
    }

    public float3 centroid()
    {
        return 0.5 * (this.pMin + this.pMax);
    }

    public float3 size()
    {
        return this.pMax - this.pMin;
    }

    public float area()
    {
        float3 size = this.size();
        return 2 * (size.x * size.y + size.y * size.z + size.z * size.x);
    }

    public float2 hit(Ray ray)
    {
        // TODO: Student implementation starts here.

        float tMin = ray.tRange.x;
        float tMax = ray.tRange.y;

        // in three directions (x, y, z)
        for (int i = 0; i < 3; i++)
        {
            // compute the range of t where the ray in the bbox for each direction
            // x(t) = o_x + t * d_x
            // t = (p_min[i] - o_x) / d_x
            float invD = 1.0 / ray.direction[i];
            float t0 = (pMin[i] - ray.origin[i]) * invD;
            float t1 = (pMax[i] - ray.origin[i]) * invD;

            // make sure t0 is "in" and t1 is "out"
            if (invD < 0.0)
            {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }

            // update the range of t
            // ray 必須 同時在 x、y、z 三個 slab 裡
            tMin = max(tMin, t0);
            tMax = min(tMax, t1);

            // no hit
            if (tMin > tMax)
            {
                return float2(1.0, 0.0); // return a fake interval
            }

        }        

        return float2(tMin, tMax);

        // TODO: Student implementation ends here.
    }
};

public interface IBoundingBox
{
    public BoundingBox boundingBox();
}

public void loadBoundingBoxes<T : IBoundingBox>(
    uint tid,
    uint offset,
    StructuredBuffer<T> primitives,
    RWStructuredBuffer<BoundingBox> boundingBoxes)
{
    uint idx = tid + offset;
    boundingBoxes[tid] = primitives[idx].boundingBox();
}

public void splitPrimitives<T : IBoundingBox>(
    uint tid,
    uint offset,
    StructuredBuffer<T> primitives,
    RWStructuredBuffer<uint> leftFlags,
    RWStructuredBuffer<uint> rightFlags,
    uint axis,
    float threshold)
{
    BoundingBox box = primitives[tid + offset].boundingBox();
    float3 centroid = box.centroid();
    if (centroid[uint(axis)] < threshold)
    {
        leftFlags[tid + offset] = 1;
        rightFlags[tid + offset] = 0;
    }
    else
    {
        leftFlags[tid + offset] = 0;
        rightFlags[tid + offset] = 1;
    }
}

public void reduceUnionPass(
    uint tid,
    uint numBoxes,
    StructuredBuffer<BoundingBox> inputBoxes,
    RWStructuredBuffer<BoundingBox> outputBoxes)
{
    uint idx = tid * 2;
    if (idx >= numBoxes)
        return;
    BoundingBox boxA = inputBoxes[idx];
    BoundingBox boxB = (idx + 1 < numBoxes) ? inputBoxes[idx + 1] : boxA;
    BoundingBox unionBox = BoundingBox.union
    (boxA, boxB);
    outputBoxes[tid] = unionBox;
}
