implementing "../renderer.slang";

import utils;

public float2 getShiftedTexUV(float2 uv, RendererUniform uniforms, Triangle triangle, float2 delta, float2 tex_uv)
{
    float2 shifted_uv = uv + delta;
    Ray ray = uniforms.camera.generateRay(shifted_uv);
    Optional<RayHitResult> hitResult = triangle.hit(ray);
    float2 result = tex_uv;
    if (hitResult.hasValue)
    {
        float3 hitPoint = ray.origin + hitResult.value.t * ray.direction;
        float3 barycentricCoord = triangle.calculateBarycentricCoord(hitPoint);
        return triangle.getUV(barycentricCoord);
    }

    Optional<RayMeshIntersectionResult> rayMeshIntersectionResult_dx = rayMeshIntersection(ray, uniforms);
    if (rayMeshIntersectionResult_dx.hasValue)
    {
        Triangle hitTriangle = rayMeshIntersectionResult_dx.value.hitTriangle;
        if (hitTriangle.getMaterialId() == triangle.getMaterialId())
        {
            float3 hitPoint = ray.origin + rayMeshIntersectionResult_dx.value.closestHit.t * ray.direction;
            float3 barycentricCoord = hitTriangle.calculateBarycentricCoord(hitPoint);
            return hitTriangle.getUV(barycentricCoord);
        }
    }

    return tex_uv;
}

public float getLevel(float2 uv, RendererUniform uniforms, Triangle triangle, float2 tex_uv)
{
    float level = 0.0;

    // TODO: Student implementation starts here.

    float2 pixelSize = 1.0 / float2(uniforms.camera.canvasSize);
    float dx = pixelSize.x;
    float dy = pixelSize.y;

    float2 uv_px = uv + float2(dx, 0.0);
    float2 uv_mx = uv - float2(dx, 0.0);
    float2 uv_py = uv + float2(0.0, dy);
    float2 uv_my = uv - float2(0.0, dy);

    Ray ray_px = uniforms.camera.generateRay(uv_px);
    Ray ray_mx = uniforms.camera.generateRay(uv_mx);
    Ray ray_py = uniforms.camera.generateRay(uv_py);
    Ray ray_my = uniforms.camera.generateRay(uv_my);

    Optional<RayMeshIntersectionResult> hit_px = rayMeshIntersection(ray_px, uniforms);
    Optional<RayMeshIntersectionResult> hit_mx = rayMeshIntersection(ray_mx, uniforms);
    Optional<RayMeshIntersectionResult> hit_py = rayMeshIntersection(ray_py, uniforms);
    Optional<RayMeshIntersectionResult> hit_my = rayMeshIntersection(ray_my, uniforms);

    // initialization
    float2 tex_uv_px = tex_uv;
    float2 tex_uv_mx = tex_uv;
    float2 tex_uv_py = tex_uv;
    float2 tex_uv_my = tex_uv;

    if (hit_px.hasValue)
    {
        Triangle tri = hit_px.value.hitTriangle;
        float3 hp = ray_px.origin + hit_px.value.closestHit.t * ray_px.direction;
        tex_uv_px = tri.getUV(tri.calculateBarycentricCoord(hp));
    }
    if (hit_mx.hasValue)
    {
        Triangle tri = hit_mx.value.hitTriangle;
        float3 hp = ray_mx.origin + hit_mx.value.closestHit.t * ray_mx.direction;
        tex_uv_mx = tri.getUV(tri.calculateBarycentricCoord(hp));
    }
    if (hit_py.hasValue)
    {
        Triangle tri = hit_py.value.hitTriangle;
        float3 hp = ray_py.origin + hit_py.value.closestHit.t * ray_py.direction;
        tex_uv_py = tri.getUV(tri.calculateBarycentricCoord(hp));
    }
    if (hit_my.hasValue)
    {
        Triangle tri = hit_my.value.hitTriangle;
        float3 hp = ray_my.origin + hit_my.value.closestHit.t * ray_my.direction;
        tex_uv_my = tri.getUV(tri.calculateBarycentricCoord(hp));
    }

    float2 duv_dx = (tex_uv_px - tex_uv_mx) / max(2.0 * dx, 1e-8);
    float2 duv_dy = (tex_uv_py - tex_uv_my) / max(2.0 * dy, 1e-8);

    float rho_x = length(duv_dx);
    float rho_y = length(duv_dy);
    float rho = max(rho_x, rho_y);

    level = max(0.0, log2(max(rho, 1e-8)));
    
    // TODO: Student implementation ends here.

    return level;
}

public PhysicsBasedParameters getHitPointParameters(float2 uv, RendererUniform uniforms, Triangle triangle, float3 normal, float3 barycentricCoord, float3 hitPoint)
{
    float2 tex_uv = triangle.getUV(barycentricCoord);
    float level = getLevel(uv, uniforms, triangle, tex_uv);
    uint materialId = triangle.materialId + triangle.offset;
    PhysicsBasedMaterial physicsBasedMaterial = uniforms.physicsBasedMaterialBuf[materialId];
    PhysicsBasedParameters parameters = sampleMaterial(physicsBasedMaterial, uniforms.physicsBasedMaterialTextureBuf, tex_uv, level, uniforms.visualizeLevelOfDetail);
    if (uniforms.visualizeBarycentricCoords)
    {
        parameters.albedo = barycentricCoord;
    }
    if (uniforms.visualizeTexUV)
    {
        parameters.albedo = float3(tex_uv, 0.0);
    }
    return parameters;
}
