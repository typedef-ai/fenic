from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class SemanticSimilarityMetric(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    COSINE: _ClassVar[SemanticSimilarityMetric]
    L2: _ClassVar[SemanticSimilarityMetric]
    DOT: _ClassVar[SemanticSimilarityMetric]

class Operator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EQ: _ClassVar[Operator]
    NOT_EQ: _ClassVar[Operator]
    LT: _ClassVar[Operator]
    LTEQ: _ClassVar[Operator]
    GT: _ClassVar[Operator]
    GTEQ: _ClassVar[Operator]
    PLUS: _ClassVar[Operator]
    MINUS: _ClassVar[Operator]
    MULTIPLY: _ClassVar[Operator]
    DIVIDE: _ClassVar[Operator]
    AND: _ClassVar[Operator]
    OR: _ClassVar[Operator]

class ChunkLengthFunction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHARACTER: _ClassVar[ChunkLengthFunction]
    WORD: _ClassVar[ChunkLengthFunction]

class ChunkCharacterSet(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CUSTOM: _ClassVar[ChunkCharacterSet]
    ASCII: _ClassVar[ChunkCharacterSet]
    UNICODE: _ClassVar[ChunkCharacterSet]
COSINE: SemanticSimilarityMetric
L2: SemanticSimilarityMetric
DOT: SemanticSimilarityMetric
EQ: Operator
NOT_EQ: Operator
LT: Operator
LTEQ: Operator
GT: Operator
GTEQ: Operator
PLUS: Operator
MINUS: Operator
MULTIPLY: Operator
DIVIDE: Operator
AND: Operator
OR: Operator
CHARACTER: ChunkLengthFunction
WORD: ChunkLengthFunction
CUSTOM: ChunkCharacterSet
ASCII: ChunkCharacterSet
UNICODE: ChunkCharacterSet
