#[derive(Clone, Eq, PartialEq, Debug)]
pub struct StateId(u64);

impl StateId {
    fn advance(&mut self) {
        self.0 += 1
    }
}

pub struct ChangeEvent {
    state: StateId,
}

impl ChangeEvent {
    pub fn new() -> Self {
        Self { state: StateId(0) }
    }

    pub fn change(&mut self) {
        self.state.advance()
    }

    pub fn register(&self) -> StateId {
        self.state.clone()
    }

    pub fn has_changed(&self, state: &mut StateId) -> bool {
        let changed = self.state.0 != state.0;
        *state = self.state.clone();
        changed
    }
}
