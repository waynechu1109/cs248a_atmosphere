implementing "../math.slang";

// A ray represented by an origin, a direction, and a t range along the ray.
public struct Ray
{
    public float3 origin;
    public float3 direction;
    // A point on the ray is given by origin + t * direction
    // tRange represents the valid interval of talong the ray: [tRange.x, tRange.y]
    public float2 tRange;

    public __init(float3 origin, float3 direction, float2 tRange = float2(0.0, float.maxValue))
    {
        this.origin = origin;
        this.direction = direction;
        this.tRange = tRange;
    }

    /**
     * Transform the ray by a 4x4 matrix.
     * @param matrix The transformation matrix.
     * @return The transformed ray.
     */
    public Ray transform(float4x4 matrix)
    {
        // TODO: Student implementation starts here.
        // float3 transformed_origin = (mul(matrix, float4(this.origin, 1.0))).xyz;
        // float3 transformed_direction = (mul(matrix, float4(this.direction, 0.0))).xyz;

        // // normalize the direction
        // transformed_direction = normalize(transformed_direction);

        // return Ray(transformed_origin, transformed_direction, this.tRange);

        float3 transformedOrigin = mul(matrix, float4(this.origin, 1.0)).xyz;
        float3 transformedDir = mul(matrix, float4(this.direction, 0.0)).xyz;
        float3 normalizedDir = normalize(transformedDir);
        return Ray(transformedOrigin, normalizedDir, this.tRange);

        // TODO: Student implementation ends here.
    }
};

public struct RayHitResult
{
    // The parameter t at which the ray intersects the primitive.
    public float t;
    // The normal of the primitive at the intersection point.
    public float3 normal;
};

public interface IRayCastable
{
    public Optional<RayHitResult> hit(Ray ray);
}
