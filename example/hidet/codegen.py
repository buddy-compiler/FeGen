import hidet

hidet.option.cache_dir('./outs/cache')

from hidet.lang import attrs, printf

with hidet.script_module() as script_module:

    # we use `hidet.script` to decorate a python function to define a hidet script function.
    @hidet.script
    def launch():
        # we use `hidet.lang.attrs` to set the attributes of the function.
        # the following line specify this hidet script function is a public function.
        attrs.func_kind = 'public'

        # print a message to the standard output.
        printf("Hello World!\n")

module = script_module.build()