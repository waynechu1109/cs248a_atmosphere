implementing "../texture.slang";

public struct SharedTexture2DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture2D tex, uint2 texelCoord)
    {
        uint2 texSize = tex.size;
        uint offset = tex.offset;
        uint index = texelCoord.y * texSize.x + texelCoord.x;
        return buffer[offset + index];
    }

    public T pointSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        if (any(uv < 0.0) || any(uv > 1.0))
        {
            return T(0.0);
        }

        int2 size = tex.size;
        float2 voxelf = uv * float2(size) - 0.5;
        int2 voxeli = clamp(int2(round(voxelf)), int2(0, 0), size - 1);

        return getBufferValue(tex, voxeli);

        // TODO: Student implementation ends here.
        return outV;
    }

    public T bilinearSample(SharedTexture2D tex, float2 uv)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.

        if (any(uv < 0.0) || any(uv > 1.0))
        {
            return T(0.0);
        }

        int2 size = tex.size;
        float2 voxelf = (uv * float2(size)) - 0.5;

        // corner of the voxel
        int2 v0 = int2(floor(voxelf));
        int2 v1 = v0 + int2(1, 1);

        v0 = clamp(v0, int2(0, 0), size - 1);
        v1 = clamp(v1, int2(0, 0), size - 1);

        // weights
        float2 w = voxelf - float2(v0);
        T wx = T(w.x);
        T wy = T(w.y);

        // sample all 4 corners
        T c00 = getBufferValue(tex, int2(v0.x, v0.y));
        T c01 = getBufferValue(tex, int2(v0.x, v1.y));
        T c10 = getBufferValue(tex, int2(v1.x, v0.y));
        T c11 = getBufferValue(tex, int2(v1.x, v1.y));

        // interpolate
        T c0 = c00 + wx * (c10 - c00);  // x direction, y = v0.y
        T c1 = c01 + wx * (c11 - c01);  // x direction, y = v1.y

        outV = c0 + wy * (c1 - c0);

        // TODO: Student implementation ends here.
        return outV;
    }

    public T trilinearSample(SharedTexture2D[] mipmapLevels, float2 uv, float d, uint totalLevels)
    {
        T outV = T(0.0);
        // TODO: Student implementation starts here.
        if (any(uv < 0.0) || any(uv > 1.0))
        {
            return T(0.0);
        }

        if (totalLevels == 0)
        {
            return T(0.0);
        }

        // "d" is mipmap level
        float level = clamp(d, 0.0, float(totalLevels - 1));
        uint level0 = (uint)floor(level);                // lower level
        uint level1 = min(level0 + 1, totalLevels - 1);  // upper level
        float w = level - float(level0);  // deceimal part -> weight

        T c0 = bilinearSample(mipmapLevels[level0], uv);
        T c1 = bilinearSample(mipmapLevels[level1], uv);
        outV = c0 + T(w) * (c1 - c0);

        // TODO: Student implementation ends here.
        return outV;
    }
}

public struct SharedTexture2D
{
    public uint2 size;
    public uint offset;
}

public struct SharedTexture3DBuffer<T>
    where T : IFloat
{
    public StructuredBuffer<T> buffer;

    T getBufferValue(SharedTexture3D tex, uint3 voxelCoord)
    {
        uint3 texSize = tex.size;
        uint offset = tex.offset;
        uint index = voxelCoord.z * texSize.y * texSize.x +
                     voxelCoord.y * texSize.x +
                     voxelCoord.x;
        return buffer[offset + index];
    }

    /**
     * Sample the texture at the given uvw coordinates using nearest neighbor interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     */
    public T pointSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (all(uvw < 0.0) || all(uvw > 1.0))
        {
            return T(0.0);
        }

        int3 size = tex.size;
        float3 voxelf = uvw * float3(size) - 0.5;
        int3 voxeli = int3(round(voxelf));

        return getBufferValue(tex, voxeli);

        // TODO: Student implementation ends here.
    }

    /**
     * Sample the texture at the given uvw coordinates using trilinear interpolation.
     * @param tex The texture to sample.
     * @param uvw The uvw coordinates to sample the texture at.
     * @return The sampled value.
     **/
    public T trilinearSample(SharedTexture3D tex, float3 uvw)
    {
        // TODO: Student implementation starts here.
        if (all(uvw < 0.0) || all(uvw > 1.0))
        {
            return T(0.0);
        }

        int3 size = tex.size;
        float3 voxelf = (uvw * float3(size)) - 0.5;

        // corner of the voxel
        int3 v0 = int3(floor(voxelf));
        int3 v1 = v0 + int3(1, 1, 1);

        // weights
        float3 w = voxelf - float3(v0);
        T wx = T(w.x);
        T wy = T(w.y);
        T wz = T(w.z);

        // sample all 8 corners
        T c000 = getBufferValue(tex, int3(v0.x, v0.y, v0.z));
        T c001 = getBufferValue(tex, int3(v0.x, v0.y, v1.z));
        T c010 = getBufferValue(tex, int3(v0.x, v1.y, v0.z));
        T c011 = getBufferValue(tex, int3(v0.x, v1.y, v1.z));
        T c100 = getBufferValue(tex, int3(v1.x, v0.y, v0.z));
        T c101 = getBufferValue(tex, int3(v1.x, v0.y, v1.z));
        T c110 = getBufferValue(tex, int3(v1.x, v1.y, v0.z));
        T c111 = getBufferValue(tex, int3(v1.x, v1.y, v1.z));

        // interpolate
        T c00 = c000 + wx * (c100 - c000);
        T c10 = c010 + wx * (c110 - c010);
        T c01 = c001 + wx * (c101 - c001);
        T c11 = c011 + wx * (c111 - c011);

        T c0 = c00 + wy * (c10 - c00);
        T c1 = c01 + wy * (c11 - c01);

        return c0 + wz * (c1 - c0);

        // TODO: Student implementation ends here.
    }
}

public struct SharedTexture3D
{
    public uint3 size;
    public uint offset;
}
