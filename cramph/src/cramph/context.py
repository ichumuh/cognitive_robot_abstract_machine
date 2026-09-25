from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Dict, Optional, Type, TypeVar

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from cramph.exceptions import (
    MissingContextExtensionError,
    DuplicateContextExtensionError,
    TickDurationUnknownError,
    ConflictingTickDurationError,
)

from semantic_digital_twin.world import World


@dataclass
class ContextExtension:
    """
    Context extension for build context.

    Used together with require_extension to augment BuildContext with custom data.
    """

    def cleanup(self):
        """
        Releases what the extension acquired while the statechart was running.
        """


GenericContextExtension = TypeVar("GenericContextExtension", bound=ContextExtension)


@dataclass
class StatechartContext:
    """
    Context handed to every node of a statechart while it is built and ticked.
    """

    world: World
    """
    The world in which the statechart is executed.
    """

    tick_duration: Optional[float] = None
    """
    How many seconds one tick stands for, None if ticks do not stand for a fixed time.
    """

    tick_variable: FloatVariable = field(init=False)
    """
    Auxiliary variable counting the ticks, can be used by nodes to implement time-
    dependent actions.
    """

    float_variable_data: FloatVariableData = field(default_factory=FloatVariableData)
    """
    Data structure used to store auxiliary variables.
    """

    extensions: Dict[Type[ContextExtension], ContextExtension] = field(
        default_factory=dict, repr=False, init=False
    )
    """
    Dictionary of extensions used to augment the build context.

    Executor extensions add the context extensions they need when an executor is
    created, see :meth:`~cramph.executor.ExecutorExtension.extend_context`.
    """

    def set_tick_duration(self, tick_duration: float):
        """
        Sets how many seconds one tick stands for.

        :param tick_duration: How many seconds one tick stands for.
        :raises ConflictingTickDurationError: If the context already knows a different
            tick duration.
        """
        if self.tick_duration is not None and self.tick_duration != tick_duration:
            raise ConflictingTickDurationError(
                tick_duration=self.tick_duration,
                requested_tick_duration=tick_duration,
            )
        self.tick_duration = tick_duration

    def require_tick_duration(self) -> float:
        """
        :return: How many seconds one tick stands for.
        :raises TickDurationUnknownError: If :attr:`tick_duration` is not known.
        """
        if self.tick_duration is None:
            raise TickDurationUnknownError()
        return self.tick_duration

    @property
    def tick_count(self) -> int:
        """
        :return: The number of ticks run since the statechart started, as held by
            :attr:`tick_variable`.
        """
        return int(self.float_variable_data.get_value(self.tick_variable))

    def require_extension(
        self, extension_type: Type[GenericContextExtension]
    ) -> GenericContextExtension:
        """
        Return an extension instance or raise ``MissingContextExtensionError``.
        """
        extension = self.extensions.get(extension_type)
        if extension is None:
            raise MissingContextExtensionError(expected_extension=extension_type)
        return extension

    def get_extension(
        self, extension_type: Type[GenericContextExtension]
    ) -> Optional[GenericContextExtension]:
        """
        :param extension_type: The exact type of the requested extension.
        :return: The extension of `extension_type`, or None if none is registered.
        """
        return self.extensions.get(extension_type)

    def add_extension(self, extension: GenericContextExtension):
        """
        Extend the build context with a custom extension.
        """
        extension_type = type(extension)
        if extension_type in self.extensions:
            raise DuplicateContextExtensionError(extension_type=extension_type)
        self.extensions[extension_type] = extension

    def cleanup(self):
        """
        Releases what the context and its extensions acquired while the statechart was
        running.
        """
        for extension in self.extensions.values():
            extension.cleanup()
