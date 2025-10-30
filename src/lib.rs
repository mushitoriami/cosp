use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::slice::Iter;
use std::str::FromStr;

#[derive(Debug, PartialEq)]
pub enum Term {
    Constant(String),
    Variable(String),
    Compound(String, Terms),
}

#[derive(Clone)]
pub struct TermsIter<'a> {
    iter: Iter<'a, Term>,
}

impl<'a> Iterator for TermsIter<'a> {
    type Item = &'a Term;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl ExactSizeIterator for TermsIter<'_> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[derive(Debug, PartialEq)]
pub struct Terms {
    vec: Vec<Term>,
}

impl From<Vec<Term>> for Terms {
    fn from(vec: Vec<Term>) -> Self {
        Terms { vec }
    }
}

impl Terms {
    fn iter(&self) -> TermsIter<'_> {
        TermsIter {
            iter: self.vec.iter(),
        }
    }
    fn len(&self) -> usize {
        self.vec.len()
    }
    fn insert(&mut self, index: usize, element: Term) {
        self.vec.insert(index, element)
    }
}

#[derive(Debug, PartialEq)]
pub enum Rule {
    Rule(u64, Term, Terms),
}

#[derive(Clone)]
pub struct RulesIter<'a> {
    iter: Iter<'a, Rule>,
}

impl<'a> Iterator for RulesIter<'a> {
    type Item = &'a Rule;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

#[derive(Debug, PartialEq)]
pub struct Rules {
    vec: Vec<Rule>,
}

impl From<Vec<Rule>> for Rules {
    fn from(vec: Vec<Rule>) -> Self {
        Rules { vec }
    }
}

impl Rules {
    fn iter(&self) -> RulesIter<'_> {
        RulesIter {
            iter: self.vec.iter(),
        }
    }
    fn insert(&mut self, index: usize, element: Rule) {
        self.vec.insert(index, element)
    }
}

fn stringify_goal(goal: (u64, &Term), table: &HashMap<(u64, &str), (u64, &Term)>) -> String {
    match goal {
        (ns, Term::Compound(label, args)) => {
            let goals_string: Vec<String> = args
                .iter()
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
        "," => {
            let mut args = take_term_args(iter)?;
            args.insert(0, term);
            Some(args)
        }
        ")" => Some(vec![term].into()),
        _ => None,
    }
}

fn take_term<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Term> {
    let label = iter.next()?;
    match iter.next()? {
        "*" => Some(Term::Constant(String::from(label))),
        "?" => Some(Term::Variable(String::from(label))),
        "(" => {
            let args = take_term_args(iter)?;
            Some(Term::Compound(String::from(label), args))
        }
        _ => None,
    }
}

fn take_terms<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Terms> {
    let term = take_term(iter)?;
    match iter.next()? {
        "," => {
            let mut args = take_terms(iter)?;
            args.insert(0, term);
            Some(args)
        }
        "." => Some(vec![term].into()),
        _ => None,
    }
}

fn take_rule<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Rule> {
    let cost = iter.next()?.parse().ok()?;
    let _ = (iter.next()? == "]").then_some(())?;
    let head = take_term(iter)?;
    match iter.next()? {
        ":-" => Some(Rule::Rule(cost, head, take_terms(iter)?.into())),
        "." => Some(Rule::Rule(cost, head, Vec::new().into())),
        _ => None,
    }
}

fn take_rules<'a>(iter: &mut impl Iterator<Item = &'a str>) -> Option<Rules> {
    match iter.next() {
        Some("[") => {
            let rule = take_rule(iter)?;
            let mut rules = take_rules(iter)?;
            rules.insert(0, rule);
            Some(rules)
        }
        None => Some(vec![].into()),
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

fn occurs_check(
    (nsv, s): (u64, &str),
    (nst, t): (u64, &Term),
    r: &HashMap<(u64, &str), (u64, &Term)>,
) -> bool {
    match t {
        Term::Compound(_, args) => args.iter().all(|c| occurs_check((nsv, s), (nst, c), r)),
        Term::Variable(s1) if nsv == nst && s == s1 => false,
        Term::Variable(s1) => r
            .get(&(nst, s1))
            .is_none_or(|&(ns1, t1)| occurs_check((nsv, s), (ns1, t1), r)),
        _ => true,
    }
}

fn unify<'a>(
    goal1: (u64, &'a Term),
    goal2: (u64, &'a Term),
    mut r: HashMap<(u64, &'a str), (u64, &'a Term)>,
) -> Option<HashMap<(u64, &'a str), (u64, &'a Term)>> {
    match (goal1, goal2) {
        ((ns1, Term::Compound(s1, args1)), (ns2, Term::Compound(s2, args2)))
            if s1 == s2 && args1.len() == args2.len() =>
        {
            let mut iter = args1.iter().zip(args2.iter());
            iter.try_fold(r, |r, (c1, c2)| unify((ns1, c1), (ns2, c2), r))
        }
        ((_, Term::Constant(s1)), (_, Term::Constant(s2))) if s1 == s2 => Some(r),
        ((ns1, Term::Variable(s1)), (ns2, Term::Variable(s2))) if ns1 == ns2 && s1 == s2 => Some(r),
        ((ns, Term::Variable(s)), goal) | (goal, (ns, Term::Variable(s)))
            if r.contains_key(&(ns, s)) =>
        {
            let &goal_variable = r.get(&(ns, s)).unwrap();
            unify(goal_variable, goal, r)
        }
        ((ns, Term::Variable(s)), goal) | (goal, (ns, Term::Variable(s)))
            if occurs_check((ns, s), goal, &r) =>
        {
            r.insert((ns, s), goal);
            Some(r)
        }
        _ => None,
    }
}

#[derive(Clone)]
struct State<'a> {
    cost: u64,
    namespace: u64,
    table: HashMap<(u64, &'a str), (u64, &'a Term)>,
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
            && goals_iter.len() == 0
        {
            shared.push((*namespace, head));
            goals.pop();
        }
    }
}

impl<'a> Iterator for Infer<'a> {
    type Item = (u64, HashMap<(u64, &'a str), (u64, &'a Term)>);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut state = self.pop_state()?;
            if self.is_empty_goal(&mut state.goals) {
                return Some((state.cost, state.table));
            }
            if let Some((namespace, term)) = state.shared_remaining.pop() {
                self.push_state(state.clone());
                let (namespace_goal, goal) = self.pop_goal(&mut state.goals);
                let Some(table) = unify((namespace, term), (namespace_goal, goal), state.table)
                else {
                    continue;
                };
                state.table = table;
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
            let Some(table) = unify((state.namespace, head), (namespace_goal, goal), state.table)
            else {
                continue;
            };
            state.table = table;
            self.push_goals(&mut state.goals, (state.namespace, head, body.iter()));
            self.update_goals(&mut state.goals, &mut state.shared);
            state.rules_iter = self.rules_iter.clone();
            state.shared_remaining = state.shared.clone();
            self.push_state(state);
        }
    }
}

fn infer_iter<'a>(goals: &'a Terms, rules: &'a Rules) -> Infer<'a> {
    let goals_iter = goals.iter();
    let rules_iter = rules.iter();
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

pub fn infer<'a>(
    goals: &'a Terms,
    rules: &'a Rules,
) -> Option<(u64, HashMap<(u64, &'a str), (u64, &'a Term)>)> {
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
                vec![
                    Term::Compound(
                        String::from("c_d"),
                        vec![Term::Constant(String::from("e_f"))].into(),
                    ),
                    Term::Variable(String::from("g_h")),
                ]
                .into(),
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
                vec![
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ]
                .into(),
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
            Ok(vec![Term::Compound(
                String::from("f"),
                vec![
                    Term::Constant(String::from("a")),
                    Term::Constant(String::from("b")),
                    Term::Variable(String::from("x")),
                ]
                .into()
            )]
            .into())
        );
    }

    #[test]
    fn test_parse_query_2() {
        let query = "f(a*, b*, x?), g(c*, y?), h(d*).".parse::<Terms>();
        assert_eq!(
            query,
            Ok(vec![
                Term::Compound(
                    String::from("f"),
                    vec![
                        Term::Constant(String::from("a")),
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("x")),
                    ]
                    .into()
                ),
                Term::Compound(
                    String::from("g"),
                    vec![
                        Term::Constant(String::from("c")),
                        Term::Variable(String::from("y")),
                    ]
                    .into()
                ),
                Term::Compound(
                    String::from("h"),
                    vec![Term::Constant(String::from("d"))].into()
                )
            ]
            .into())
        );
    }

    #[test]
    fn test_parse_rules_1() {
        let rules = "[2]a* :- b*, c?.   \n[4]d*.\n".parse::<Rules>();
        assert_eq!(
            rules,
            Ok(vec![
                Rule::Rule(
                    2,
                    Term::Constant(String::from("a")),
                    vec![
                        Term::Constant(String::from("b")),
                        Term::Variable(String::from("c"))
                    ]
                    .into()
                ),
                Rule::Rule(4, Term::Constant(String::from("d")), vec![].into())
            ]
            .into())
        );
    }

    #[test]
    fn test_take_rules_2() {
        let rules = "[2]a:-b,C.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_take_rules_3() {
        let rules = "[2]a:-b,C.  \n[4]d.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_take_rules_4() {
        let rules = "[2]*a :- *b, ?c.   \n[4]*d.\n".parse::<Rules>();
        assert_eq!(rules, Err(()));
    }

    #[test]
    fn test_unify_1() {
        assert_eq!(
            unify(
                (0, &"f(a* ,b* ,x? )".parse().unwrap()),
                (1, &"f(y? ,b* ,c* )".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((0, "x"), (1, &"c*".parse().unwrap())),
                ((1, "y"), (0, &"a*".parse().unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_2() {
        assert_eq!(
            unify(
                (1, &"f(x? ,y? )".parse().unwrap()),
                (1, &"f(a* ,b* )".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((1, "x"), (1, &"a*".parse().unwrap())),
                ((1, "y"), (1, &"b*".parse().unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_3() {
        assert_eq!(
            unify(
                (0, &"x?".parse().unwrap()),
                (0, &"y?".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (0, &"y?".parse().unwrap()))])
        )
    }

    #[test]
    fn test_unify_4() {
        assert_eq!(
            unify(
                (0, &"f(a*,b*)".parse().unwrap()),
                (1, &"f(x?,x?)".parse().unwrap()),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_5() {
        assert_eq!(
            unify(
                (0, &"x?".parse().unwrap()),
                (0, &"f(x?)".parse().unwrap()),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_6() {
        assert_eq!(
            unify(
                (1, &"f(f(x?),g(y?))".parse().unwrap()),
                (1, &"f(y?,x?)".parse().unwrap()),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_7() {
        assert_eq!(
            unify(
                (1, &"g(x?,y?,x?)".parse().unwrap()),
                (1, &"g(f(x?),f(y?),y?)".parse().unwrap()),
                HashMap::new()
            ),
            None
        )
    }

    #[test]
    fn test_unify_8() {
        assert_eq!(
            unify(
                (0, &"x?".parse().unwrap()),
                (0, &"x?".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::new()
        )
    }

    #[test]
    fn test_unify_9() {
        assert_eq!(
            unify(
                (0, &"x?".parse().unwrap()),
                (1, &"f(x?)".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (1, &"f(x?)".parse().unwrap()))])
        )
    }

    #[test]
    fn test_unify_10() {
        assert_eq!(
            unify(
                (0, &"x?".parse().unwrap()),
                (1, &"x?".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([((0, "x"), (1, &"x?".parse().unwrap()))])
        )
    }

    #[test]
    fn test_unify_11() {
        assert_eq!(
            unify(
                (0, &"f(f(x?),g(y?))".parse().unwrap()),
                (1, &"f(y?,x?)".parse().unwrap()),
                HashMap::new()
            )
            .unwrap(),
            HashMap::from([
                ((1, "x"), (0, &"g(y?)".parse().unwrap())),
                ((1, "y"), (0, &"f(x?)".parse().unwrap()))
            ])
        )
    }

    #[test]
    fn test_unify_12() {
        assert_eq!(
            unify(
                (0, &"f(f(x?), x?)".parse().unwrap()),
                (1, &"f(x?,x?)".parse().unwrap()),
                HashMap::new()
            ),
            None
        )
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(0, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_1_2() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(liz*, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_1_3() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(tom*, ben*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_1_4() {
        let rules = &RULES1.parse().unwrap();
        let query = &"parent(x?, liz*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
        let res =
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>();
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
        let res =
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>();
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
        let res =
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>();
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(
                0,
                HashMap::from([
                    ((1, "x"), (0, &"tom*".parse().unwrap())),
                    ((1, "z"), (0, &"pat*".parse().unwrap())),
                    ((1, "y"), (2, &"bob*".parse().unwrap())),
                    ((3, "x"), (2, &"bob*".parse().unwrap())),
                    ((3, "z"), (0, &"pat*".parse().unwrap()))
                ])
            )]
        )
    }

    #[test]
    fn test_infer_3_2() {
        let rules = &RULES3.parse().unwrap();
        let query = &"parent(pam*, bob*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(0, HashMap::from([]))]
        )
    }

    #[test]
    fn test_infer_3_3() {
        let rules = &RULES3.parse().unwrap();
        let query = &"mother(pam*, bob*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [(2, HashMap::new())]
        );
    }

    #[test]
    fn test_infer_4_2() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(liz*, pat*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_4_3() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(tom*, ben*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            []
        );
    }

    #[test]
    fn test_infer_4_4() {
        let rules = &RULES4.parse().unwrap();
        let query = &"parent(x?, liz*).".parse().unwrap();
        assert_eq!(
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
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
            infer_iter(query, rules).collect::<Vec<(u64, HashMap<(u64, &str), (u64, &Term)>)>>(),
            [
                (3, HashMap::from([((0, "x"), (1, &"p*".parse().unwrap())),])),
                (5, HashMap::from([((0, "x"), (1, &"p*".parse().unwrap())),])),
                (6, HashMap::from([((0, "x"), (1, &"q*".parse().unwrap())),]))
            ]
        )
    }
}
