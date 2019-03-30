from lupa import LuaError
from lua.sandbox import LuaSandbox


def assert_lua_error(lua, code):
    try:
        lua.execute(code)
    except LuaError:
        assert True
        return
    raise AssertionError(f'Lua code {code!r} should error')


def _test_sandbox(*, allow_global_state):

    lua = LuaSandbox(
        allow_global_state=allow_global_state,
    )

    if allow_global_state:
        # Setting globals should be allowed:
        global_set_test = lua.execute
    else:
        # Setting globals should be blocked (nil):
        global_set_test = lambda code: assert_lua_error(lua, code)

    # These should all be accessible:
    assert lua.eval('tonumber')
    assert lua.eval('string.format')
    assert lua.eval('os.date')

    # These should all be blocked (nil):
    assert not lua.eval('not_a_real_global')  # not a real global
    assert not lua.eval('io')  # real global
    assert not lua.eval('os.execute')

    # Setting local variables should always be allowed.
    lua.execute('local some_local = 10')
    lua.execute('''
        local some_local = 10
        some_local = 20
    ''')
    lua.execute('''
        local some_local = 10
        local function f(x)
            local some_other_local = x
            local some_local = x
        end
    ''')

    # Accessing variables from within closures should always be allowed.
    lua.execute('''
        local some_local = 10
        local function f(t)
            t.key = some_local
        end
    ''')

    # Testing globals
    global_set_test('some_global = 10')
    global_set_test('string.k = 10')
    global_set_test('string = 10')
    global_set_test('function global_func(x) return x end')

    # Testing closures [NYI]
    # global_set_test('''
    #     local some_local = 10
    #     local function f(x)
    #         some_local = x
    #     end
    # ''')
    # global_set_test('''
    #     local some_local = {}
    #     local function f(x)
    #         some_local.key = x
    #     end
    # ''')

def test_sandbox_allow_set_global():
    return _test_sandbox(allow_global_state=True)

def test_sandbox_no_allow_set_global():
    return _test_sandbox(allow_global_state=False)
