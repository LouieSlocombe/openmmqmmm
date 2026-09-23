**Title:** Fix OpenMMTheory.set_numcores() so it updates the CPU thread configuration

Suggested labels: `bug`  
Suggested priority: High

Calling `OpenMMTheory.set_numcores(n)` changes `self.numcores`, but new OpenMM Contexts use `self.properties["Threads"]`, which retains the constructor value. `QMMMTheory.set_numcores()` forwards to this setter, so its advertised MM resource change does not take effect.

For an existing CPU theory constructed with one thread, calling `mm.set_numcores(4)` leaves `mm.properties["Threads"] == "1"`.

Update the CPU platform property together with the public core count. Define how changes interact with an already-created Context; do not silently claim to reconfigure an active Context.

Acceptance criteria:

- [ ] Calling the setter before Context creation changes the actual CPU `Threads` property.
- [ ] Calling through `QMMMTheory.set_numcores()` updates the nested MM configuration.
- [ ] Existing Context behavior is documented and either updated safely or rejected explicitly.
- [ ] Non-CPU platforms are not assigned unsupported CPU properties.
- [ ] Regression coverage verifies the effective Context property, not only `self.numcores`.

Source: [thread setter](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/theory.py#L892), [platform configuration](https://github.com/LouieSlocombe/openmmqmmm/blob/b4495f805e29b210e26ef978b481f2883a07cbcc/openmmqmmm/openmm/theory.py#L335).

