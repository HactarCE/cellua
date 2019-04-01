import lupa
import sys

from utils import resources

with open(resources.get('VERSION')) as version_file:
    VERSION = version_file.read().strip()

lua = lupa.LuaRuntime()
FULL_LUA_VERSION = LUA_VERSION = lua.eval('_VERSION')
LUAJIT_VERSION = lua.eval('jit and jit.version')
if LUAJIT_VERSION:
    FULL_LUA_VERSION += f' ({LUAJIT_VERSION})'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-V', '--version', help="display version info", action='store_true')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    args = parser.parse_args()
    if args.version:
        print(f"Cellua {VERSION}")
        print(f"Python {sys.version.split()[0]}")
        print(f"{FULL_LUA_VERSION}")
