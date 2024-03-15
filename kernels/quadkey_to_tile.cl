__kernel void quadkey_to_tile(__global const int *quadkeys, __global int *tiles, const int quadkey_length)
{
    int gid = get_global_id(0);
    int tileX = 0;
    int tileY = 0;
    int tileZ = quadkey_length;

    for (int i = 0; i < quadkey_length; i++)
    {
        tileZ--;

        int mask = 1 << i;
        int qk_digit = quadkeys[gid * quadkey_length + i];

        if (qk_digit == 1)
            tileX |= mask;
        else if (qk_digit == 2)
            tileY |= mask;
        else if (qk_digit == 3)
        {
            tileX |= mask;
            tileY |= mask;
        }
    }

    tiles[gid * 3 + 0] = tileX;
    tiles[gid * 3 + 1] = tileY;
    tiles[gid * 3 + 2] = tileZ;
}