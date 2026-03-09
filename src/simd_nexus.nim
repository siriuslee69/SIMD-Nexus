#============================================#
# simd_nexus.nim                         #
# <- Module aggregator for SIMD helpers.     #
#============================================#

import 
    ./simd_nexus/simd/base_operations,
    ./simd_nexus/simd/converters,
    ./simd_nexus/simd/custom_operations,
    ./simd_nexus/simd/generic_u32,
    ./simd_nexus/simd/generic_i8,
    ./simd_nexus/simd/generic_i16,
    ./simd_nexus/simd/generic_u64,
    ./simd_nexus/simd/generic_f32,
    ./simd_nexus/simd/generic_f64,
    ./simd_nexus/simd/iterators,
    ./simd_nexus/sequences/custom_operations as sequence_custom_operations,
    ./simd_nexus/sequences/byte_streams as sequence_byte_streams

export
    base_operations,
    converters,
    custom_operations,
    generic_u32,
    generic_i8,
    generic_i16,
    generic_u64,
    generic_f32,
    generic_f64,
    iterators,
    sequence_custom_operations,
    sequence_byte_streams

