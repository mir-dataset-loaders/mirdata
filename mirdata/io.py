import functools
import io
from typing import BinaryIO, Callable, Optional, TextIO, TypeVar, Union

T = TypeVar("T")  # Can be anything


def coerce_to_string_io(
    func: Callable[[TextIO], T]
) -> Callable[[Optional[Union[str, TextIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, TextIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj) as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.StringIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper


def coerce_to_bytes_io(
    func: Callable[[BinaryIO], T]
) -> Callable[[Optional[Union[str, BinaryIO]]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(file_path_or_obj: Optional[Union[str, BinaryIO]]) -> Optional[T]:
        if not file_path_or_obj:
            return None
        if isinstance(file_path_or_obj, str):
            with open(file_path_or_obj, "rb") as f:
                return func(f)
        elif isinstance(file_path_or_obj, io.BytesIO):
            return func(file_path_or_obj)
        else:
            raise ValueError(
                "Invalid argument passed to {}, argument has the type {}",
                func.__name__,
                type(file_path_or_obj),
            )

    return wrapper
