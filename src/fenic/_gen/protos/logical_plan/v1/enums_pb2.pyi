from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

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
    TOKEN: _ClassVar[ChunkLengthFunction]

class ChunkCharacterSet(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CUSTOM: _ClassVar[ChunkCharacterSet]
    ASCII: _ClassVar[ChunkCharacterSet]
    UNICODE: _ClassVar[ChunkCharacterSet]

class JoinType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INNER: _ClassVar[JoinType]
    FULL: _ClassVar[JoinType]
    LEFT: _ClassVar[JoinType]
    RIGHT: _ClassVar[JoinType]
    CROSS: _ClassVar[JoinType]

class FuzzySimilarityMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    INDEL: _ClassVar[FuzzySimilarityMethod]
    LEVENSHTEIN: _ClassVar[FuzzySimilarityMethod]
    DAMERAU_LEVENSHTEIN: _ClassVar[FuzzySimilarityMethod]
    JARO_WINKLER: _ClassVar[FuzzySimilarityMethod]
    JARO: _ClassVar[FuzzySimilarityMethod]
    HAMMING: _ClassVar[FuzzySimilarityMethod]

class DocContentType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MARKDOWN: _ClassVar[DocContentType]
    JSON: _ClassVar[DocContentType]
    PDF: _ClassVar[DocContentType]
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
TOKEN: ChunkLengthFunction
CUSTOM: ChunkCharacterSet
ASCII: ChunkCharacterSet
UNICODE: ChunkCharacterSet
INNER: JoinType
FULL: JoinType
LEFT: JoinType
RIGHT: JoinType
CROSS: JoinType
INDEL: FuzzySimilarityMethod
LEVENSHTEIN: FuzzySimilarityMethod
DAMERAU_LEVENSHTEIN: FuzzySimilarityMethod
JARO_WINKLER: FuzzySimilarityMethod
JARO: FuzzySimilarityMethod
HAMMING: FuzzySimilarityMethod
MARKDOWN: DocContentType
JSON: DocContentType
PDF: DocContentType
