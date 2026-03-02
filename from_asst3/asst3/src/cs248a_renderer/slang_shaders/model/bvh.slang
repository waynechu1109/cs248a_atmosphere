implementing "../model.slang";
import math;

public struct BVHNode
{
    public BoundingBox bound;
    public int left;
    public int right;
    public int primLeft;
    public int primRight;
    public int depth;

    public __init(BoundingBox bound, int left = -1, int right = -1, int primLeft = -1, int primRight = -1, int depth = -1)
    {
        this.bound = bound;
        this.left = left;
        this.right = right;
        this.primLeft = primLeft;
        this.primRight = primRight;
        this.depth = depth;
    }

    public float2 hit(Ray ray)
    {
        return this.bound.hit(ray);
    }
}

public struct BVH<T>
    where T : IRayCastable
{
    public struct HitResult
    {
        public T primitive;
        public RayHitResult rayHitResult;
    }

    // Slang buffer of BVHNode copied from Python.
    public RWStructuredBuffer<BVHNode> nodes;
    // Maximum number of nodes in the BVH, in this case, it'll be the size of the nodes buffer.
    public uint maxNodes;
    // Slang buffer of primitives represented by the BVH.
    public RWStructuredBuffer<T> primitives;
    // Number of primitives in the primitives buffer.
    public uint numPrimitives;

    public __init(RWStructuredBuffer<BVHNode> nodes, RWStructuredBuffer<T> primitives, uint numPrimitives, uint maxNodes)
    {
        this.nodes = nodes;
        this.primitives = primitives;
        this.numPrimitives = numPrimitives;
        this.maxNodes = maxNodes;
    }

    struct HitStack<int N>
    {
        uint[N] nodeIndices;
        uint top;

        __init()
        {
            this.top = 0;
        }

        [mutating]
        void push(uint nodeIndex)
        {
            this.nodeIndices[this.top] = nodeIndex;
            this.top += 1;
        }

        [mutating]
        uint pop()
        {
            this.top -= 1;
            return this.nodeIndices[this.top];
        }

        bool isEmpty()
        {
            return this.top == 0;
        }

        uint size()
        {
            return this.top;
        }
    }

    // BVH traversal to find the closest hit.
    public Optional<HitResult> hit(Ray ray)
    {
        // Initialize traversal.
        Optional<HitResult> hitResult = Optional<HitResult>();
        HitStack<64> stack;

        // TODO: Student implementation starts here.

        float closestT = ray.tRange.y;
        stack.push(0); // root node index = 0, starts from node index 0, cannot do recursion on GPU

        while (!stack.isEmpty())
        {
            uint nodeIdx = stack.pop();
            if (nodeIdx >= this.maxNodes)
                continue;

            BVHNode node = this.nodes[nodeIdx];      // the node we are working on now

            // 1. ray vs bbox
            float2 tRange = node.hit(ray);
            if (tRange.x > tRange.y || tRange.y < ray.tRange.x || tRange.x > closestT)
                // a) ray 根本沒打到 bbox
                // b) ray 起點已經超過 bbox
                // c) bbox 比目前最近 hit 還遠，沒必要再看
                continue;

            // -1 if this is a leaf node.
            bool isLeaf = (node.left < 0 && node.right < 0);   // leaf: 真的存放 primitive index 範圍

            if (isLeaf)
            {
                // 2. leaf: test primitives
                for (int i = node.primLeft; i < node.primRight; i++)
                {
                    T prim = this.primitives[i];
                    Optional<RayHitResult> primHit = prim.hit(ray);

                    if (primHit.hasValue)
                    {
                        float t = primHit.value.t;
                        if (t >= ray.tRange.x && t < closestT)
                        {
                            closestT = t;
                            HitResult hr;
                            hr.primitive = prim;
                            hr.rayHitResult = primHit.value;
                            hitResult = Optional<HitResult>(hr);
                        }
                    }
                }
            }
            else
            {
                // 3. interior node: push children
                if (node.left >= 0)
                    stack.push(node.left);
                if (node.right >= 0)
                    stack.push(node.right);
            }
        }

        // TODO: Student implementation ends here.

        return hitResult;
    }
}
