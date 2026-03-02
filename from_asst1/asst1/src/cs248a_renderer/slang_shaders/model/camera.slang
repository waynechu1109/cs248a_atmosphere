implementing "../model.slang";
import math;

public struct Camera
{
    // The inverse view matrix of the camera (matrix transform from view space to world space).
    public float4x4 invViewMatrix;
    // The canvas size of the target image in pixels.
    public uint2 canvasSize;
    // The focal length of the camera in pixels (distance between the pinhole and the image plane).
    public float focalLength;

    /**
     * Generate a camera ray given a normalized uv coordinate on canvas.
     * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
     * @return The generated Ray. Before you start working the transform section, generate ray as if the camera is at origin looking towards negative z direction.
     * After you finish the transform section, you should transform the ray from view space to world space using the inverse view matrix.
     */
    public Ray generateRay(float2 uv)
    {
        // TODO: Student implementation starts here.

        // Convert uv to pixel coordinates centered at (0,0)
        // float px = (uv.x - 0.5f) * canvasSize.x;
        // float py = (uv.y - 0.5f) * canvasSize.y;

        // float3 direction = normalize(float3(px, py, -focalLength));
        // float3 origin = float3(0, 0, 0);

        // float3 originWorld =
        //     mul(invViewMatrix, float4(origin, 1.0)).xyz;
        // float3 dirWorld =
        //     normalize(mul(invViewMatrix, float4(direction, 0.0)).xyz);

        // return Ray(originWorld, dirWorld, float2(0.0, float.maxValue));

        // // convert uv to pixel coordinates centered at (0,0)
        // float px = (uv.x - 0.5f) * canvasSize.x;
        // float py = (uv.y - 0.5f) * canvasSize.y;

        // // camera at (0, 0, 0) looking towards negative z direction
        // // 從相機原點指向螢幕上該像素的方向
        // float3 direction = normalize(float3(px, py, -focalLength));

        // Ray rayView = Ray(float3(0,0,0), direction);

        // // transform the ray from view space to world space
        // return rayView.transform(invViewMatrix);

        float px = (float(uv.x) - 0.5) * this.canvasSize.x;
        float py = (float(uv.y) - 0.5) * this.canvasSize.y;

        float3 viewDir = float3(px, py, -this.focalLength);

        Ray rayView = Ray(float3(0.0, 0.0, 0.0), normalize(viewDir));

        return rayView.transform(this.invViewMatrix);
        // TODO: Student implementation ends here.
    }
}
