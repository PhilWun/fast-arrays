# Fast Arrays

This library can be used to perform vectorized calculations on arbitrarily sized 1D and 2D arrays.
It uses `AVX-512 F` operations, but can also use standard operations as fallback.
To compile a Rust project with `AVX-512 F` you need to run the compiler with `-Ctarget-feature=+avx512f` or use a config file like [config.toml](.cargo/config.toml).
