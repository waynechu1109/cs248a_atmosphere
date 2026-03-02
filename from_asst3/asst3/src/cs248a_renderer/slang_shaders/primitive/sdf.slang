implementing "../primitive.slang";
import math;

public struct SDFBuffer
{
    public StructuredBuffer<SphereSDF> spheres;
    public StructuredBuffer<CubeSDF> cubes;
    public uint sphereCount;
    public uint cubeCount;
    public CustomSDF customSDF;
    public bool renderCustomSDF;

    /**
     * Compute the minimum signed distance from a point to all SDF primitives in the buffer.
     * @param point The point from which to compute the signed distance.
     * @return A Tuple containing the minimum signed distance and the normal of the closest SDF primitive.
     */
    public Tuple<float, float3> sample(float3 point)
    {
        float minDist = float.maxValue;
        float3 closestNormal = float3(0.0, 0.0, 0.0);
        for (uint i = 0; i < this.sphereCount; i++)
        {
            float dist = this.spheres[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.spheres[i].normal(point);
            }
        }
        for (uint i = 0; i < this.cubeCount; i++)
        {
            float dist = this.cubes[i].sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.cubes[i].normal(point);
            }
        }
        if (this.renderCustomSDF)
        {
            float dist = this.customSDF.sdf(point);
            if (dist < minDist)
            {
                minDist = dist;
                closestNormal = this.customSDF.normal(point);
            }
        }
        return Tuple<float, float3>(minDist, closestNormal);
    }
}

// Compose SDFs together using union, intersection, and difference operations
[Differentiable]
public float sdfUnion(float d1, float d2)
{
    return min(d1, d2);
}

[Differentiable]
public float sdfIntersection(float d1, float d2)
{
    return max(d1, d2);
}

[Differentiable]
public float sdfDifference(float d1, float d2)
{
    return max(d1, -d2);
}

public interface ISDF
{
    /**
     * Compute the signed distance from a po˝int to the surface of the primitive.
     * @param point The point from which to compute the signed distance.
     * @return The signed distance from the point to the surface of the primitive.
     */
    public float sdf(float3 point);

    /**
     * The normal of the SDF at a given point.
     * @param point The point at which to compute the normal.
     * @return The normal vector at the given point.
     */
    public float3 normal(float3 point);
}

public struct SphereSDF : ISDF
{
    public float3 center;
    public float radius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.

        // // transform the point from world space to local space
        // float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;

        // // return SDF value
        // return length(pLocal - this.center) - this.radius;

        float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        return length(pLocal - this.center) - this.radius;

        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.

        // // transform the point from world space to local space
        // float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;

        // // compute normal in local space
        // float3 nLocal = normalize(pLocal - this.center);

        // // transform local normal back to world space (use inverse-transpose)
        // // since normal is not position vector, we have to use inverse-transpose
        // float3 nWorld = mul(transpose(invModelMatrix), float4(nLocal, 0.0)).xyz;
        // return nWorld;

        float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        float3 nLocal = normalize(pLocal - this.center);
        float3 nWorld = mul(transpose(this.invModelMatrix), float4(pLocal, 0.0)).xyz;

        return nWorld;        

        // TODO: Student implementation ends here.
    }
}

public struct CubeSDF : ISDF
{
    public float3 size;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        // TODO: Student implementation starts here.

        // // transform the point from world space to local space
        // float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;

        // // get the half size of the cube
        // float3 halfSize = this.size * 0.5;

        // // map the point to first octant (cube is symmetric to every axis)
        // float3 q = abs(pLocal) - halfSize;

        // float outsideDist = length(max(q, float3(0.0, 0.0, 0.0)));

        // // max(q.x, max(q.y, q.z)) 找出最接近 0 的軸
        // float insideDist = min(max(q.x, max(q.y, q.z)), 0.0);

        // return outsideDist + insideDist;

        float3 pLocal = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        float3 halfSize = this.size * 0.5;
        float3 q = abs(point) - halfSize;

        float outDist = length(max(q, float3(0.0, 0.0, 0.0)));
        float inDIst = min(max(q.x, max(q.y, q.z)), 0.0);

        return outDist + inDIst;

        // TODO: Student implementation ends here.
    }

    public float3 normal(float3 point)
    {
        // TODO: Student implementation starts here.

        // // directly compute in world space
        // // the gradient of SDF is normal
        // const float eps = 1e-4;

        // float dx = sdf(point + float3(eps, 0, 0)) - sdf(point - float3(eps, 0, 0));
        // float dy = sdf(point + float3(0, eps, 0)) - sdf(point - float3(0, eps, 0));
        // float dz = sdf(point + float3(0, 0, eps)) - sdf(point - float3(0, 0, eps));

        // return normalize(float3(dx, dy, dz));
        float eps = 1e-4;

        float dx = this.sdf(point + float3(eps, 0.0, 0.0)) - this.sdf(point - float3(eps, 0.0, 0.0));
        float dy = this.sdf(point + float3(0.0, eps, 0.0)) - this.sdf(point - float3(0.0, eps, 0.0));
        float dz = this.sdf(point + float3(0.0, 0.0, eps)) - this.sdf(point - float3(0.0, 0.0, eps));

        return normalize(float3(dx, dy, dz));

        // TODO: Student implementation ends here.
    }
}

public struct CustomSDF : ISDF
{
    public float3 cubeSize;
    public float sphereRadius;
    public float4x4 invModelMatrix;

    [NoDiffThis, Differentiable]
    public float sdf(float3 point)
    {
        float3 localPoint = mul(this.invModelMatrix, float4(point, 1.0)).xyz;
        CubeSDF outerCube;
        outerCube.size = this.cubeSize;
        outerCube.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, 0.0),
            float4(0.0, 1.0, 0.0, 0.0),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0)); // Identity matrix
        SphereSDF innerSphere;
        innerSphere.center = float3(0.0, 0.0, 0.0);
        innerSphere.radius = this.sphereRadius;
        innerSphere.invModelMatrix = float4x4(
            float4(1.0, 0.0, 0.0, -0.5),
            float4(0.0, 1.0, 0.0, 0.2),
            float4(0.0, 0.0, 1.0, 0.0),
            float4(0.0, 0.0, 0.0, 1.0)); // Identity matrix
        float outerDist = outerCube.sdf(localPoint);
        float innerDist = innerSphere.sdf(localPoint);
        return sdfDifference(outerDist, innerDist);
    }

    [Differentiable]
    static float sdfHelper(no_diff CustomSDF sdf, float3 point)
    {
        return sdf.sdf(point);
    }

    public float3 normal(float3 point)
    {
        DifferentialPair<float3> dp = diffPair(point);
        bwd_diff(CustomSDF.sdfHelper)(this, dp, 1.0);
        return normalize(dp.d);
    }
}
