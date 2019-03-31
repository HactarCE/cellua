def make_table_readonly(lua, tbl, error_format="Cannot set value '%s' on %s"):
    """Wrap a Lua table with a metatable that prevents writes."""
    return lua.eval('''
        function(tbl, error_format)
            local new_table = {}
            setmetatable(new_table, {
                __index=tbl,
                __newindex=function(t, k, v)
                    error(string.format(error_format, k, new_table))
                end,
            })
            return new_table
        end
    ''')(tbl, error_format)

def make_table_readonly_recursive(lua, tbl, *args, **kwargs):
    """Run LuaSandbox.make_table_readonly() recursively, so that any tables
    that are members of this one are also read-only.
    """
    new_table = lua.table()
    for key, value in tbl.items():
        if lua.eval('type')(value) == 'table':
            new_table[key] = make_table_readonly_recursive(lua, value, *args, **kwargs)
        else:
            new_table[key] = value
    return make_table_readonly(lua, new_table, *args, **kwargs)
