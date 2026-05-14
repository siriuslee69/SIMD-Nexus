#============================================#
# simd_nexus.nim                         #
# <- Module aggregator for SIMD helpers.     #
#============================================#

import 
    ./protocols/simd/base_operations,
    ./protocols/simd/converters,
    ./protocols/simd/custom_operations,
    ./protocols/simd/generic_u32,
    ./protocols/simd/generic_i32,
    ./protocols/simd/generic_i8,
    ./protocols/simd/generic_i16,
    ./protocols/simd/generic_u64,
    ./protocols/simd/generic_f32,
    ./protocols/simd/generic_f64,
    ./protocols/simd/iterators,
    ./protocols/sequences/custom_operations as sequence_custom_operations,
    ./protocols/sequences/byte_streams as sequence_byte_streams,
    ./protocols/gpu/dispatch as gpu_dispatch

export
    base_operations,
    converters,
    custom_operations,
    generic_u32,
    generic_i32,
    generic_i8,
    generic_i16,
    generic_u64,
    generic_f32,
    generic_f64,
    iterators,
    sequence_custom_operations,
    sequence_byte_streams,
    gpu_dispatch

