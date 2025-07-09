from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class SemanticSimilarityMetric(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COSINE: _ClassVar[SemanticSimilarityMetric]
    L2: _ClassVar[SemanticSimilarityMetric]
    DOT: _ClassVar[SemanticSimilarityMetric]

class OperatorProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EQ: _ClassVar[OperatorProto]
    NOT_EQ: _ClassVar[OperatorProto]
    LT: _ClassVar[OperatorProto]
    LTEQ: _ClassVar[OperatorProto]
    GT: _ClassVar[OperatorProto]
    GTEQ: _ClassVar[OperatorProto]
    PLUS: _ClassVar[OperatorProto]
    MINUS: _ClassVar[OperatorProto]
    MULTIPLY: _ClassVar[OperatorProto]
    DIVIDE: _ClassVar[OperatorProto]
    AND: _ClassVar[OperatorProto]
    OR: _ClassVar[OperatorProto]

class ChunkLengthFunctionProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHARACTER: _ClassVar[ChunkLengthFunctionProto]
    WORD: _ClassVar[ChunkLengthFunctionProto]

class ChunkCharacterSetProto(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CUSTOM: _ClassVar[ChunkCharacterSetProto]
    ASCII: _ClassVar[ChunkCharacterSetProto]
    UNICODE: _ClassVar[ChunkCharacterSetProto]
COSINE: SemanticSimilarityMetric
L2: SemanticSimilarityMetric
DOT: SemanticSimilarityMetric
EQ: OperatorProto
NOT_EQ: OperatorProto
LT: OperatorProto
LTEQ: OperatorProto
GT: OperatorProto
GTEQ: OperatorProto
PLUS: OperatorProto
MINUS: OperatorProto
MULTIPLY: OperatorProto
DIVIDE: OperatorProto
AND: OperatorProto
OR: OperatorProto
CHARACTER: ChunkLengthFunctionProto
WORD: ChunkLengthFunctionProto
CUSTOM: ChunkCharacterSetProto
ASCII: ChunkCharacterSetProto
UNICODE: ChunkCharacterSetProto
