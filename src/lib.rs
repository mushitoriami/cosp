use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::str::FromStr;

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Term {
    Constant(String),
    Variable(String),
    Compound(String, Terms),
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum Rule {
    Rule(u64, Term, Terms),
}

#[derive(Debug)]
#[cfg_attr(test, derive(PartialEq))]
pub enum List<T> {
    Empty(),
    Cons(Box<T>, Box<List<T>>),
}

type Terms = List<Term>;
type TermsIter<'a> = &'a Terms;
type Rules = List<Rule>;
type RulesIter<'a> = &'a Rules;

impl<'a, T> Iterator for &'a List<T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let List::Cons(term, terms) = self else {
            return None;
        };
        *self = terms;
        Some(term)
    }
}

impl<T> FromIterator<T> for List<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
        let mut iter = iterable.into_iter();
        match iter.next() {
            Some(term) => List::head_and_tail(term, iter.collect()),
            None => List::new(),
        }
    }
}

impl<T> List<T> {
    fn new() -> Self {
        List::Empty()
    }
    fn head_and_tail(head: T, tail: Self) -> Self {
        List::Cons(Box::new(head), Box::new(tail))
    }
    fn len(&self) -> usize {
        match self {
            List::Empty() => 0,
            List::Cons(_, terms) => terms.len() + 1,
        }
    }
    fn is_empty(&self) -> bool {
        match self {
            List::Empty() => true,
            List::Cons(_, _) => false,
        }
    }
}

fn stringify_goal(goal: (u64, &Term), table: &HashMap<(u64, &str), (u64, &Term)>) -> String {
    match goal {
        (ns, Term::Compound(label, args)) => {
            let goals_string: Vec<String> = args
                .into_iter()
                .map(|x| stringify_goal((ns, x), table))
                .collect();
            label.clone() + "(" + &goals_string.join(", ") + ")"
        }
        (_, Term::Constant(label)) => label.clone() + "*",
        (ns, Term::Variable(label)) => match table.get(&(ns, label)) {
            Some(&goal) => stringify_goal(goal, table),
            None => label.clone() + "#" + &ns.to_string(),
        },
    }
}

pub fn stringify_table(table: &HashMap<(u64, &str), (u64, &Term)>) -> Vec<String> {
    let mut res = Vec::new();
    for (&(ns, label), &goal) in table {
        if ns == 0 {
            res.push(label.to_string() + " = " + &stringify_goal(goal, table) + "\n");
        }
    }
    res
}

fn take_term_args<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Terms> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => Some(Terms::head_and_tail(term, take_term_args(iter)?)),
        ")" => Some(Terms::head_and_tail(term, Terms::new())),
        _ => None,
    }
}

fn take_term<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Term> {
    let label = iter.next()?;
    match iter.next()? {
        "*" => Some(Term::Constant(String::from(label))),
        "?" => Some(Term::Variable(String::from(label))),
        "(" => Some(Term::Compound(String::from(label), take_term_args(iter)?)),
        _ => None,
    }
}

fn take_terms<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Terms> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => Some(Terms::head_and_tail(term, take_terms(iter)?)),
        "." => Some(Terms::head_and_tail(term, Terms::new())),
        _ => None,
    }
}

fn take_rule<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Rule> {
    let cost = iter.next()?.parse().ok()?;
    let _ = (iter.next()? == "]").then_some(())?;
    let head = take_term(iter)?;
    match iter.next()? {
        ":-" => Some(Rule::Rule(cost, head, take_terms(iter)?)),
        "." => Some(Rule::Rule(cost, head, Terms::new())),
        _ => None,
    }
}

fn take_rules<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Rules> {
    match iter.next() {
        Some("[") => Some(Rules::head_and_tail(take_rule(iter)?, take_rules(iter)?)),
        None => Some(Rules::new()),
        _ => None,
    }
}

impl FromStr for Term {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?"]);
        let mut iter = tokenizer.tokenize(s).map_while(|x| x.ok());
        let term = take_term(&mut iter).ok_or(())?;
        iter.next().is_none().then_some(term).ok_or(())
    }
}

impl FromStr for Terms {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?", "."]);
        let mut iter = tokenizer.tokenize(s).map_while(|x| x.ok());
        let query = take_terms(&mut iter).ok_or(())?;
        iter.next().is_none().then_some(query).ok_or(())
    }
}

impl FromStr for Rules {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut tokenizer = kohaku::Tokenizer::new(["(", ")", ",", "*", "?", ".", ":-", "[", "]"]);
        let mut iter = tokenizer.tokenize(s).map_while(|x| x.ok());
        let rules = take_rules(&mut iter).ok_or(())?;
        iter.next().is_none().then_some(rules).ok_or(())
    }
}

type Table<'a> = HashMap<(u64, &'a str), (u64, &'a Term)>;

fn variables(t: &Term) -> Vec<&str> {
    match t {
        Term::Constant(_) => Vec::new(),
        Term::Variable(s) => [s.as_str()].into(),
        Term::Compound(_, args) => args.into_iter().flat_map(|x| variables(x)).collect(),
    }
}

fn occurs_check((nsv, s): (u64, &str), (nst, t): (u64, &Term), r: &Table) -> bool {
    variables(t).into_iter().all(|s1| match r.get(&(nst, s1)) {
        Some(&g) => occurs_check((nsv, s), g, r),
        None => (nst, s1) != (nsv, s),
    })
}

fn matchings_terms<'a>(ts1: &'a Terms, ts2: &'a Terms) -> Option<Vec<(&'a Term, &'a Term)>> {
    (ts1.len() == ts2.len()).then_some(())?;
    let mut res = Vec::new();
    for (c1, c2) in ts1.into_iter().zip(ts2.into_iter()) {
        res.extend(matchings(c1, c2)?);
    }
    Some(res)
}

fn matchings<'a>(t1: &'a Term, t2: &'a Term) -> Option<Vec<(&'a Term, &'a Term)>> {
    match (t1, t2) {
        (Term::Constant(s1), Term::Constant(s2)) if s1 == s2 => Some(Vec::new()),
        (Term::Variable(_), _) | (_, Term::Variable(_)) => Some(vec![(t1, t2)]),
        (Term::Compound(s1, ts1), Term::Compound(s2, ts2)) if s1 == s2 => matchings_terms(ts1, ts2),
        _ => None,
    }
}

fn add_matching<'a>(goal1: (u64, &'a str), goal2: (u64, &'a Term), r: &mut Table<'a>) -> bool {
    match r.get(&goal1) {
        Some(&goal) => unify(goal, goal2, r),
        None => occurs_check(goal1, goal2, r) && r.insert(goal1, goal2).is_none(),
    }
}

fn unify<'a>(goal1: (u64, &'a Term), goal2: (u64, &'a Term), r: &mut Table<'a>) -> bool {
    match matchings(goal1.1, goal2.1) {
        Some(m) => m.into_iter().all(|x| match x {
            (Term::Variable(s1), Term::Variable(s2)) if (goal1.0, s1) == (goal2.0, s2) => true,
            (Term::Variable(s), t) => add_matching((goal1.0, s), (goal2.0, t), r),
            (t, Term::Variable(s)) => add_matching((goal2.0, s), (goal1.0, t), r),
            _ => unreachable!(),
        }),
        None => false,
    }
}

#[derive(Clone)]
struct State<'a> {
    cost: u64,
    namespace: u64,
    table: Table<'a>,
    shared: Vec<(u64, &'a Term)>,
    shared_remaining: Vec<(u64, &'a Term)>,
    goals: Vec<(u64, &'a Term, TermsIter<'a>)>,
    rules_iter: RulesIter<'a>,
}

impl Eq for State<'_> {}

impl PartialEq for State<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl PartialOrd for State<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cost.cmp(&other.cost)
    }
}

struct Infer<'a> {
    rules_iter: RulesIter<'a>,
    pq: BinaryHeap<Reverse<State<'a>>>,
}

impl<'a> Infer<'a> {
    fn push_state(&mut self, state: State<'a>) {
        self.pq.push(Reverse(state))
    }

    fn pop_state(&mut self) -> Option<State<'a>> {
        self.pq.pop().map(|x| x.0)
    }

    fn push_goals(
        &mut self,
        goals: &mut Vec<(u64, &'a Term, TermsIter<'a>)>,
        goals_iter: (u64, &'a Term, TermsIter<'a>),
    ) {
        goals.push(goals_iter)
    }

    fn pop_goal(&mut self, goals: &mut Vec<(u64, &'a Term, TermsIter<'a>)>) -> (u64, &'a Term) {
        let (namespace, _, iter) = goals.last_mut().unwrap();
        (*namespace, iter.next().unwrap())
    }

    fn is_empty_goal(&mut self, goals: &mut Vec<(u64, &'a Term, TermsIter<'a>)>) -> bool {
        goals.is_empty()
    }

    fn update_goals(
        &mut self,
        goals: &mut Vec<(u64, &'a Term, TermsIter<'a>)>,
        shared: &mut Vec<(u64, &'a Term)>,
    ) {
        while let Some((namespace, head, goals_iter)) = goals.last_mut()
            && goals_iter.is_empty()
        {
            shared.push((*namespace, head));
            goals.pop();
        }
    }
}

impl<'a> Iterator for Infer<'a> {
    type Item = (u64, Table<'a>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut state = self.pop_state()?;
            if self.is_empty_goal(&mut state.goals) {
                return Some((state.cost, state.table));
            }
            if let Some((namespace, term)) = state.shared_remaining.pop() {
                self.push_state(state.clone());
                let (namespace_goal, goal) = self.pop_goal(&mut state.goals);
                if !unify((namespace, term), (namespace_goal, goal), &mut state.table) {
                    continue;
                };
                self.update_goals(&mut state.goals, &mut state.shared);
                state.rules_iter = self.rules_iter.clone();
                state.shared_remaining = state.shared.clone();
                self.push_state(state);
                continue;
            }
            let Some(Rule::Rule(cost_rule, head, body)) = state.rules_iter.next() else {
                continue;
            };
            self.push_state(state.clone());
            let (namespace_goal, goal) = self.pop_goal(&mut state.goals);
            state.cost = state.cost + cost_rule;
            state.namespace += 1;
            if !unify(
                (state.namespace, head),
                (namespace_goal, goal),
                &mut state.table,
            ) {
                continue;
            };
            self.push_goals(&mut state.goals, (state.namespace, head, body.into_iter()));
            self.update_goals(&mut state.goals, &mut state.shared);
            state.rules_iter = self.rules_iter.clone();
            state.shared_remaining = state.shared.clone();
            self.push_state(state);
        }
    }
}

fn infer_iter<'a>(goals: &'a Terms, rules: &'a Rules) -> Infer<'a> {
    let goals_iter = goals.into_iter();
    let rules_iter = rules.into_iter();
    Infer {
        rules_iter: rules_iter.clone(),
        pq: BinaryHeap::from([Reverse(State {
            cost: 0,
            namespace: 0,
            table: HashMap::new(),
            shared: Vec::new(),
            shared_remaining: Vec::new(),
            goals: vec![(0, goals_iter.clone().next().unwrap(), goals_iter.clone())],
            rules_iter: rules_iter.clone(),
        })]),
    }
}

pub fn infer<'a>(goals: &'a Terms, rules: &'a Rules) -> Option<(u64, Table<'a>)> {
    infer_iter(goals, rules).next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stringify_goal_1() {
        assert_eq!(
            stringify_goal((0, &"a*".parse().unwrap()), &HashMap::new()),
            "a*"
        );
    }

    #[test]
    fn test_stringify_goal_2() {
        assert_eq!(
            stringify_goal((0, &"x?".parse().unwrap()), &HashMap::new()),
            "x#0"
        );
    }

    #[test]
    fn test_stringify_goal_3() {
        assert_eq!(
            stringify_goal(
                (2, &"x?".parse().unwrap()),
                &HashMap::from([((2, "x"), (1, &"y?".parse().unwrap()))])
            ),
            "y#1"
        );
    }

    #[test]
    fn test_stringify_goal_4() {
        assert_eq!(
            stringify_goal((0, &"ab(c_d(e_f*),g_h?)".parse().unwrap()), &HashMap::new()),
            "ab(c_d(e_f*), g_h#0)"
        );
    }

    #[test]
    fn test_stringify_goal_5() {
        assert_eq!(
            stringify_goal(
                (2, &"f(a*, b*, x?)".parse().unwrap()),
                &HashMap::from([((2, "x"), (1, &"ab(c_d(e_f*),g_h?)".parse().unwrap()))])
            ),
            "f(a*, b*, ab(c_d(e_f*), g_h#1))"
        );
    }

    #[test]
    fn test_stringify_table_1() {
        let strings = stringify_table(&HashMap::from([
            ((0, "x"), (1, &"x?".parse().unwrap())),
            ((1, "x"), (2, &"x?".parse().unwrap())),
            ((0, "y"), (1, &"x?".parse().unwrap())),
        ]));
        assert_eq!(strings.len(), 2);
        assert!(strings.contains(&"x = x#2\n".into()));
        assert!(strings.contains(&"y = x#2\n".into()));
    }

    #[test]
    fn test_parse_term_1() {
        assert_eq!(
            "ab(c_d(e_f*),g_h?)".parse(),
            Ok(Term::Compound(
                String::from("ab"),
                Terms::from_iter([
                    Term::Compound(
                        String::from("c_d"),
                        Terms::from_iter([Term::Constant(String::from("e_f"))]),
                    ),
                    Term::Variable(String::from("g_h")),
                ]),
            ))
        );
    }

    #[test]
    fn test_parse_term_2() {
        assert_eq!("ab(c_d(*e_f),?g_h)))(".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_3() {
        assert_eq!("a*(".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_4() {
        assert_eq!("a,a*)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_5() {
        assert_eq!("a(b*(c*)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_6() {
        assert_eq!("a)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_7() {
        assert_eq!("a(b**".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_8() {
        assert_eq!("a(*)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_9() {
        assert_eq!("(a*)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_10() {
        assert_eq!("a*a".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_11() {
        assert_eq!("a(a*a)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_12() {
        assert_eq!(
            "f(a*, b*, x?)".parse(),
            Ok(Term::Compound(
                String::from("f"),
                Terms::from_iter([
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ]),
            ))
        );
    }

    #[test]
    fn test_parse_term_13() {
        assert_eq!("f(a, b, X)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_14() {
        assert_eq!("f(*a, *b, ?x)".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_15() {
        assert_eq!("ab(c_d(e_f),g_h)))(".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_term_16() {
        assert_eq!("ab(c_d(e_f*),g_h?)))(".parse::<Term>(), Err(()));
    }

    #[test]
    fn test_parse_query_1() {
        let query = "f(a*, b*, x?).".parse::<Terms>();
        assert_eq!(
            query,
            Ok(Terms::from_iter([Term::Compound(
                String::from("f"),
                Terms::from_iter([
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ])
            )]))
        );
    }

    #[test]
    fn test_parse_query_2() {
        let query = "f(a*, b*, x?), g(c*, y?), h(d*).".parse::<Terms>();
        assert_eq!(
            query,
            Ok(Terms::from_iter([
                Term::Compound(
                    String::from("f"),
                    Terms::from_iter([
                        Term::Constant(String::from("a")),
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("x")),
                    ])
                ),
                Term::Compound(
                    String::from("g"),
                    Terms::from_iter([
                        Term::Constant(String::from("c")),
                        Term::Variable(String::from("y")),
                    ])
                ),
                Term::Compound(
                    String::from("h"),
                    Terms::from_iter([Term::Constant(String::from("d"))])
                )
            ]))
        );
    }

    #[test]
    fn test_parse_rules_1() {
        let rules = "[2]a* :- b*, c?.   \n[4]d*.\n".parse::<Rules>();
        assert_eq!(
            rules,
            Ok(Rules::from_iter([
                Rule::Rule(
                    2,
                    Term::Constant(String::from("a")),
                    Terms::from_iter([
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("c"))
                    ])
                ),
                Rule::Rule(4, Term::Constant(String::from("d")), Terms::new())
            ]))
        );
    }

    #[test]
    fn test_parse_rules_2() {
        let rules = "[2]a:-b,C.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_parse_rules_3() {
        let rules = "[2]a:-b,C.  \n[4]d.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_parse_rules_4() {
        let rules = "[2]*a :- *b, ?c.   \n[4]*d.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_unify_1() {
        let term1 = "f(a* ,b* ,x? )".parse().unwrap();
        let term2 = "f(y? ,b* ,c* )".parse().unwrap();
        let term3 = "c*".parse().unwrap();
        let term4 = "a*".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (1, &term2), &mut table));
        assert_eq!(
            table,
            [((0, "x"), (1, &term3)), ((1, "y"), (0, &term4))].into()
        );
    }

    #[test]
    fn test_unify_2() {
        let term1 = "f(x? ,y? )".parse().unwrap();
        let term2 = "f(a* ,b* )".parse().unwrap();
        let term3 = "a*".parse().unwrap();
        let term4 = "b*".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((1, &term1), (1, &term2), &mut table));
        assert_eq!(
            table,
            [((1, "x"), (1, &term3)), ((1, "y"), (1, &term4))].into()
        );
    }

    #[test]
    fn test_unify_3() {
        let term1 = "x?".parse().unwrap();
        let term2 = "y?".parse().unwrap();
        let term3 = "y?".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (0, &term2), &mut table));
        assert_eq!(table, [((0, "x"), (0, &term3))].into());
    }

    #[test]
    fn test_unify_4() {
        let term1 = "f(a*,b*)".parse().unwrap();
        let term2 = "f(x?,x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(!unify((0, &term1), (1, &term2), &mut table));
    }

    #[test]
    fn test_unify_5() {
        let term1 = "x?".parse().unwrap();
        let term2 = "f(x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(!unify((0, &term1), (0, &term2), &mut table));
    }

    #[test]
    fn test_unify_6() {
        let term1 = "f(f(x?),g(y?))".parse().unwrap();
        let term2 = "f(y?,x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(!unify((1, &term1), (1, &term2), &mut table));
    }

    #[test]
    fn test_unify_7() {
        let term1 = "g(x?,y?,x?)".parse().unwrap();
        let term2 = "g(f(x?),f(y?),y?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(!unify((1, &term1), (1, &term2), &mut table));
    }

    #[test]
    fn test_unify_8() {
        let term1 = "x?".parse().unwrap();
        let term2 = "x?".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (0, &term2), &mut table));
        assert_eq!(table, HashMap::new());
    }

    #[test]
    fn test_unify_9() {
        let term1 = "x?".parse().unwrap();
        let term2 = "f(x?)".parse().unwrap();
        let term3 = "f(x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (1, &term2), &mut table));
        assert_eq!(table, [((0, "x"), (1, &term3))].into());
    }

    #[test]
    fn test_unify_10() {
        let term1 = "x?".parse().unwrap();
        let term2 = "x?".parse().unwrap();
        let term3 = "x?".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (1, &term2), &mut table));
        assert_eq!(table, [((0, "x"), (1, &term3))].into());
    }

    #[test]
    fn test_unify_11() {
        let term1 = "f(f(x?),g(y?))".parse().unwrap();
        let term2 = "f(y?,x?)".parse().unwrap();
        let term3 = "g(y?)".parse().unwrap();
        let term4 = "f(x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(unify((0, &term1), (1, &term2), &mut table));
        assert_eq!(
            table,
            [((1, "x"), (0, &term3)), ((1, "y"), (0, &term4))].into()
        );
    }

    #[test]
    fn test_unify_12() {
        let term1 = "f(f(x?), x?)".parse().unwrap();
        let term2 = "f(x?,x?)".parse().unwrap();
        let mut table = HashMap::new();
        assert!(!unify((0, &term1), (1, &term2), &mut table));
    }

    const RULES1: &str = r#"
        [0] parent(pam*, bob*).
        [0] parent(tom*, bob*).
        [0] parent(tom*, liz*).
        [0] parent(bob*, ann*).
        [0] parent(bob*, pat*).
        [0] parent(pat*, jim*).
    "#;

    #[test]
    fn test_infer_1_1() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(bob*, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(0, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_1_2() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(liz*, pat*).".parse().unwrap();
        assert_eq!(infer_iter(query, rules).collect::<Vec<(u64, Table)>>(), []);
    }

    #[test]
    fn test_infer_1_3() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(tom*, ben*).".parse().unwrap();
        assert_eq!(infer_iter(query, rules).collect::<Vec<(u64, Table)>>(), []);
    }

    #[test]
    fn test_infer_1_4() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(x?, liz*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([((0, "x"), (1, &"tom*".parse().unwrap()))])
            )]
        );
    }

    #[test]
    fn test_infer_1_5() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(bob*, y?).".parse().unwrap();
        let res = infer_iter(query, rules).collect::<Vec<(u64, Table)>>();
        assert!(res.len() == 2);
        assert!(res.contains(&(
            0,
            HashMap::from([((0, "y"), (1, &"ann*".parse().unwrap()))])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([((0, "y"), (1, &"pat*".parse().unwrap()))])
        )));
    }

    #[test]
    fn test_infer_1_6() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(p?, q?).".parse().unwrap();
        let res = infer_iter(query, rules).collect::<Vec<(u64, Table)>>();
        assert!(res.len() == 6);
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"pam*".parse().unwrap())),
                ((0, "q"), (1, &"bob*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"tom*".parse().unwrap())),
                ((0, "q"), (1, &"bob*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"tom*".parse().unwrap())),
                ((0, "q"), (1, &"liz*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"bob*".parse().unwrap())),
                ((0, "q"), (1, &"ann*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"bob*".parse().unwrap())),
                ((0, "q"), (1, &"pat*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "p"), (1, &"pat*".parse().unwrap())),
                ((0, "q"), (1, &"jim*".parse().unwrap()))
            ])
        )));
    }

    #[test]
    fn test_infer_1_7() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(y?, jim*), parent(x?, y?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((0, "y"), (1, &"pat*".parse().unwrap())),
                    ((0, "x"), (2, &"bob*".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_1_8() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(tom*, x?), parent(x?, y?).".parse().unwrap();
        let res = infer_iter(query, rules).collect::<Vec<(u64, Table)>>();
        assert!(res.len() == 2);
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "x"), (1, &"bob*".parse().unwrap())),
                ((0, "y"), (2, &"ann*".parse().unwrap()))
            ])
        )));
        assert!(res.contains(&(
            0,
            HashMap::from([
                ((0, "x"), (1, &"bob*".parse().unwrap())),
                ((0, "y"), (2, &"pat*".parse().unwrap()))
            ])
        )));
    }

    #[test]
    fn test_infer_1_9() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(x?, ann*), parent(x?, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([((0, "x"), (1, &"bob*".parse().unwrap()))])
            )]
        )
    }

    #[test]
    fn test_infer_1_10() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(pam*, x?), parent(x?, y?), parent(y?, jim*)."
            .parse()
            .unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((0, "x"), (1, &"bob*".parse().unwrap())),
                    ((0, "y"), (2, &"pat*".parse().unwrap()))
                ])
            )]
        )
    }

    const RULES2: &str = r#"
        [0] big(bear*).
        [0] big(elephant*).
        [0] small(cat*).
        [0] brown(bear*).
        [0] black(cat*).
        [0] gray(elephant*).
        [0] dark(z?) :- black(z?).
        [0] dark(z?) :- brown(z?).
    "#;

    #[test]
    fn test_infer_2() {
        let rules = &RULES2.parse().unwrap();
        let query = &"dark(x?), big(x?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "z"), (0, &"x?".parse().unwrap())),
                    ((0, "x"), (2, &"bear*".parse().unwrap()))
                ])
            )]
        )
    }

    const RULES3: &str = r#"
        [0] parent(pam*, bob*).
        [0] parent(tom*, bob*).
        [0] parent(tom*, liz*).
        [0] parent(bob*, ann*).
        [0] parent(bob*, pat*).
        [0] parent(pat*, jim*).
        [0] female(pam*).
        [0] male(tom*).
        [0] male(bob*).
        [0] female(liz*).
        [0] female(ann*).
        [0] female(pat*).
        [0] male(jim*).
        [0] offspring(y?, x?) :- parent(x?, y?).
        [0] mother(x?, y?) :- parent(x?, y?), female(x?).
        [0] grandparent(x?, z?) :- parent(x?, y?), parent(y?, z?).
        [0] sister(x?, y?) :- parent(z?, x?), parent(z?, y?), female(x?), different(x?, y?).
        [0] predecessor(x?, z?) :- parent(x?, z?).
        [0] predecessor(x?, z?) :- parent(x?, y?), predecessor(y?, z?).
    "#;

    #[test]
    fn test_infer_3_1() {
        let rules = &RULES3.parse().unwrap();
        let query = &"predecessor(tom*, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &"tom*".parse().unwrap())),
                    ((1, "z"), (0, &"pat*".parse().unwrap())),
                    ((1, "y"), (2, &"bob*".parse().unwrap())),
                    ((3, "x"), (1, &"y?".parse().unwrap())),
                    ((3, "z"), (1, &"z?".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_2() {
        let rules = &RULES3.parse().unwrap();
        let query = &"parent(pam*, bob*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(0, HashMap::from([]))]
        )
    }

    #[test]
    fn test_infer_3_3() {
        let rules = &RULES3.parse().unwrap();
        let query = &"mother(pam*, bob*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &"pam*".parse().unwrap())),
                    ((1, "y"), (0, &"bob*".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_4() {
        let rules = &RULES3.parse().unwrap();
        let query = &"grandparent(pam*, ann*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &"pam*".parse().unwrap())),
                    ((1, "z"), (0, &"ann*".parse().unwrap())),
                    ((1, "y"), (2, &"bob*".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_5() {
        let rules = &RULES3.parse().unwrap();
        let query = &"grandparent(bob*, jim*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &"bob*".parse().unwrap())),
                    ((1, "z"), (0, &"jim*".parse().unwrap())),
                    ((1, "y"), (2, &"pat*".parse().unwrap()))
                ])
            )]
        )
    }

    const RULES4: &str = r#"
        [6] parent(pam*, bob*).
        [5] parent(tom*, bob*).
        [4] parent(tom*, liz*).
        [3] parent(bob*, ann*).
        [2] parent(bob*, pat*).
        [1] parent(pat*, jim*).
    "#;

    #[test]
    fn test_infer_4_1() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(bob*, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(2, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_4_2() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(liz*, pat*).".parse().unwrap();
        assert_eq!(infer_iter(query, rules).collect::<Vec<(u64, Table)>>(), []);
    }

    #[test]
    fn test_infer_4_3() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(tom*, ben*).".parse().unwrap();
        assert_eq!(infer_iter(query, rules).collect::<Vec<(u64, Table)>>(), []);
    }

    #[test]
    fn test_infer_4_4() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(x?, liz*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                4,
                HashMap::from([((0, "x"), (1, &"tom*".parse().unwrap()))])
            )]
        );
    }

    #[test]
    fn test_infer_4_5() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(bob*, y?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [
                (
                    2,
                    HashMap::from([((0, "y"), (1, &"pat*".parse().unwrap()))])
                ),
                (
                    3,
                    HashMap::from([((0, "y"), (1, &"ann*".parse().unwrap()))])
                )
            ]
        );
    }

    #[test]
    fn test_infer_4_6() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(p?, q?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [
                (
                    1,
                    HashMap::from([
                        ((0, "p"), (1, &"pat*".parse().unwrap())),
                        ((0, "q"), (1, &"jim*".parse().unwrap()))
                    ])
                ),
                (
                    2,
                    HashMap::from([
                        ((0, "p"), (1, &"bob*".parse().unwrap())),
                        ((0, "q"), (1, &"pat*".parse().unwrap()))
                    ])
                ),
                (
                    3,
                    HashMap::from([
                        ((0, "p"), (1, &"bob*".parse().unwrap())),
                        ((0, "q"), (1, &"ann*".parse().unwrap()))
                    ])
                ),
                (
                    4,
                    HashMap::from([
                        ((0, "p"), (1, &"tom*".parse().unwrap())),
                        ((0, "q"), (1, &"liz*".parse().unwrap()))
                    ])
                ),
                (
                    5,
                    HashMap::from([
                        ((0, "p"), (1, &"tom*".parse().unwrap())),
                        ((0, "q"), (1, &"bob*".parse().unwrap()))
                    ])
                ),
                (
                    6,
                    HashMap::from([
                        ((0, "p"), (1, &"pam*".parse().unwrap())),
                        ((0, "q"), (1, &"bob*".parse().unwrap()))
                    ])
                )
            ]
        );
    }

    #[test]
    fn test_infer_4_7() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(y?, jim*), parent(x?, y?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                1 + 2,
                HashMap::from([
                    ((0, "y"), (1, &"pat*".parse().unwrap())),
                    ((0, "x"), (2, &"bob*".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_4_8() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(tom*, x?), parent(x?, y?).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [
                (
                    5 + 2,
                    HashMap::from([
                        ((0, "x"), (1, &"bob*".parse().unwrap())),
                        ((0, "y"), (2, &"pat*".parse().unwrap()))
                    ])
                ),
                (
                    5 + 3,
                    HashMap::from([
                        ((0, "x"), (1, &"bob*".parse().unwrap())),
                        ((0, "y"), (2, &"ann*".parse().unwrap()))
                    ])
                )
            ]
        )
    }

    #[test]
    fn test_infer_4_9() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(x?, ann*), parent(x?, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                3 + 2,
                HashMap::from([((0, "x"), (1, &"bob*".parse().unwrap()))])
            )]
        )
    }

    #[test]
    fn test_infer_4_10() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(pam*, x?), parent(x?, y?), parent(y?, jim*)."
            .parse()
            .unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(
                6 + 2 + 1,
                HashMap::from([
                    ((0, "x"), (1, &"bob*".parse().unwrap())),
                    ((0, "y"), (2, &"pat*".parse().unwrap()))
                ])
            )]
        )
    }

    const RULES5: &str = r#"
        [2] p* :- q*.
        [1] q*.
        [1] p* :- r*.
        [3] r*.
    "#;

    #[test]
    fn test_infer_5_1() {
        let rules = &RULES5.parse().unwrap();
        let query = &"p*.".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [(3, HashMap::from([])), (4, HashMap::from([]))]
        )
    }

    const RULES6: &str = r#"
        [1] f(p*) :- g(q*).
        [2] g(q*).
        [4] f(q*).
    "#;

    #[test]
    fn test_infer_6_1() {
        let rules = &RULES6.parse().unwrap();
        let query = &"f(x?), g(q*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, Table)>>(),
            [
                (3, HashMap::from([((0, "x"), (1, &"p*".parse().unwrap())),])),
                (5, HashMap::from([((0, "x"), (1, &"p*".parse().unwrap())),])),
                (6, HashMap::from([((0, "x"), (1, &"q*".parse().unwrap())),]))
            ]
        )
    }
}
