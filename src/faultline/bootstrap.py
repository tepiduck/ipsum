"""Load this harness in another interpreter without borrowing its dependencies."""

import os
from pathlib import Path

_BOOTSTRAP = """import importlib.util, sys
package = sys.argv.pop(1)
program = sys.argv.pop(1)
spec = importlib.util.spec_from_file_location(
    'faultline', package, submodule_search_locations=[str(__import__('pathlib').Path(package).parent)]
)
module = importlib.util.module_from_spec(spec)
sys.modules['faultline'] = module
spec.loader.exec_module(module)
exec(compile(program, '<faultline-bootstrap>', 'exec'), {'__name__': '__main__'})
"""


def python_command(python, program, arguments=(), *, watch_parent=False):
    """Keep the child interpreter's dependency search path intact.

    Adding the coordinator's package parent to PYTHONPATH works in a source tree,
    but from a wheel that parent is site-packages and can replace the child's
    PyTorch. Loading only our package also supports workers without Faultline
    installed, while resolving their torch and other dependencies normally.
    """
    if watch_parent:
        from .runner import process_identity

        parent = (os.getpid(), process_identity(os.getpid()))
        program = (
            "import threading\nfrom faultline.worker import watch_parent\n"
            f"threading.Thread(target=watch_parent,args={parent!r},daemon=True).start()\n" + program
        )
    return [
        str(python),
        "-c",
        _BOOTSTRAP,
        str(Path(__file__).resolve().with_name("__init__.py")),
        program,
        *map(str, arguments),
    ]
