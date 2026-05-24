/* Sample functions for RL-UCO tests */

static int helper(int x) { return x * 2 + 1; }

int sum_loop(int n) {
    int s = 0;
    for (int i = 0; i < n; i++)
        s += helper(i);
    return s;
}

int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int dot(const int *a, const int *b, int n) {
    int s = 0;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
}
