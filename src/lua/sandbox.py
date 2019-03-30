from lupa import LuaRuntime
import lupa


# http://lua-users.org/wiki/SandBoxes
# Some variables that are listed on the Lua wiki as safe are still forbidden for
# use by automata:
# - coroutine.* -- Python will handle the multithreading, thank you very much.
# - math.random -- Automata should be deterministic (for now).
# - os.clock, os.difftime, and os.time -- Automata should not depend on the
#   system clock.

LUA_SAFE_NAMES = list(filter(None, map(str.strip, '''

assert error

ipairs next pairs pcall print rawequal select tonumber tostring type unpack
_VERSION xpcall

string.byte string.char string.find string.format string.gmatch string.gsub
string.len string.lower string.match string.rep string.reverse string.sub
string.upper

table.concat table.insert table.maxn table.remove table.sort

math.abs math.acos math.atan math.atan2 math.ceil math.cos math.cosh math.deg
math.exp math.floor math.fmod math.floor math.fmod math.frexp math.huge
math.ldexp math.log math.log10 math.max math.min math.modf math.pi math.pow
math.rad math.sin math.sinh math.sqrt math.tan math.tanh

'''.split())))

LUA_RANDOM_NAMES = ['math.random']
LUA_TIME_NAMES = ['os.clock', 'os.date', 'os.difftime', 'os.time']


class LuaSandbox:
    """A wrapper for LuaRuntime to use when running untrusted code.

    This has some Cellua-specific features, like scanning for unauthorized use
    of globals or closures [closures NYI].
    """

    def __init__(self, *,
                 allow_global_state=True, # TODO not yet fully implemented
                 allow_random=True,
                 allow_time=True,
                 custom_globals=None,
                 **kwargs):
        """Initialize a sandboxed LuaRuntime.

        Optional keyword arguments:
        - `allow_global_state` -- bool; whether to allow setting global
          variables and setting external variables from within a closure
          [closures NYI]
        - `allow_random` -- bool; whether to give access to Lua's PRNG functions
        - `allow_time` -- bool; whether to give access to Lua's date/time
          functions
        - `custom_globals` -- dict; any extra variables to insert into the
          global namespace

        All other args and kwargs are passed to LuaRuntime(). It is not
        necessary to pass `register_eval=False` or `register_builtins=False`,
        since this is already done by LuaSandbox.
        """
        self._allow_global_state = allow_global_state
        # Prevent access to Python from Lua.
        self._lua = LuaRuntime(register_eval=False, register_builtins=False, **kwargs)
        # Get the behind-the-scenes global table.
        allowed_names = LUA_SAFE_NAMES
        if allow_random:
            allowed_names += LUA_RANDOM_NAMES
        if allow_time:
            allowed_names += LUA_TIME_NAMES
        new_globals = {}
        for name in allowed_names:
            # We have to handle names like `string.format` by making a new table
            # `string` containing keys like `format`.
            t = new_globals
            keys = name.split('.')
            for key in keys[:-1]:
                if key not in t:
                    t[key] = self.table()
                t = t[key]
            t[keys[-1]] = self._lua.eval(name)
        if custom_globals:
            new_globals.update(custom_globals)
        new_globals = self.table_from(new_globals)
        # Override global table access if necessary.
        if not allow_global_state:
            new_globals = self.make_table_readonly_recursive(
                new_globals,
                "Cannot set value '%s' on %s; global variables are forbidden"
            )
        self.globals().safe_globals = new_globals
        if self.globals().setfenv:
            # Lua 5.1
            self._sandboxer_code = 'setfenv(1, safe_globals)\n'
        else:
            # Lua 5.2+
            self._sandboxer_code = '_ENV = safe_globals\n'

    def __getattr__(self, name):
        return getattr(self._lua, name)

    def make_table_readonly(self, table, error_format="Cannot set value '%s' on %s"):
        """Wrap a Lua table with a metatable that prevents writes."""
        return self._lua.eval('''
            function(tab, error_format)
                local new_table = {}
                setmetatable(new_table, {
                    __index=tab,
                    __newindex=function(t, k, v)
                        error(string.format(error_format, k, new_table))
                    end,
                })
                return new_table
            end
        ''')(table, error_format)

    def make_table_readonly_recursive(self, table, *args, **kwargs):
        """Run LuaSandbox.make_table_readonly() recursively, so that any tables
        that are members of this one are also read-only.
        """
        new_table = self.table()
        for key, value in table.items():
            if self._lua.eval('type')(value) == 'table':
                new_table[key] = self.make_table_readonly_recursive(value, *args, **kwargs)
            else:
                new_table[key] = value
        return self.make_table_readonly(new_table, *args, **kwargs)

    def _sandbox(self, lua_code):
        # TODO: Handle self._allow_global_state here by scanning AST for
        # closures. Raise an exception if one is found.
        print(self.globals()._VERSION)
        print(self._sandboxer_code)
        print(repr(self._sandboxer_code + lua_code))
        return self._sandboxer_code + lua_code

    def compile(self, lua_code):
        return self._lua.compile(self._sandbox(lua_code))

    def eval(self, lua_code, *args):
        return self._lua.execute(self._sandbox('return ' + lua_code), *args)

    def execute(self, lua_code, *args):
        return self._lua.execute(self._sandbox(lua_code), *args)

    def require(self, lua_code, *args):
        raise NotImplemented(f"{self.__class__.__name__}.require() is not yet implemented.")
