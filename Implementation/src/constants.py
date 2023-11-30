BUG_CATEGORY_COLUMN_NAME = "Campo personalizado (Categoria do Bug)"

COLUMNS_TO_REMOVE = [
    'Ambiente\(s\)',
    'Anexo',
    'Approvers',
    'Atualizado',
    'Ação do Risco',
    'Categoria do status',
    'Categorias',
    'Chave da item',
    'Chave do projeto',
    'Checklist Progress',
    'Comentar',
    'Comentar',
    'Componentes',
    'Criado',
    'Criador',
    'Data de Conclusão',
    'Date of First Response',
    'Descrição do projeto',
    'Development',
    'Driver de Qualidade',
    'Flagged',
    'Frequência',
    'ICE Score',
    'ID',
    'Impacto do Bug/Melhoria',
    'Link',
    'Líder do projeto',
    'Linha de Negócio',
    'Nome do projeto',
    'Número NFs',
    'Observadores',
    'PR Review',
    'Pai',
    'Parent summary',
    'Prioridade',
    'Produto',
    'Pull Requests',
    'Rank',
    'Relator',
    'Request Type',
    'Request language',
    'Resolução',
    'Resolvido',
    'Responsável',
    'Revisor',
    'Severidade',
    'Solução',
    'Sprint',
    'Status',
    'Story points',
    'Tester',
    'Time',
    'Tipo de item',
    'Tipo de projeto',
    'Tipo de teste',
    'URL',
    'Versão de lançamento',
    'Versão de lançamento',
    'Versão do App',
    'Versão do App',
    'Versões corrigidas',
    'Votos',
    'Work category',
    'Última visualização'
]

COLUMNS_WITH_SAME_WORDS_AS_REMAINING_COLUMNS = [
    "Descrição"
]

ROOT_CAUSE_CLASSIFICATION = {
    'Construção': [
      'C2: Erro no ambiente ou arquitetura (Libs desatualizadas)',
      'C5: Erro de implementação de componente',
      'C8: Erro de implementação de constraints de banco',
      'C9: Requisito/regra de negócio não atendida/observada',
      'C10: Não adesão ao requisito/regra de negócio definida',
      'C11: Erro de lógica na integração de diferentes códigos',
      'C12: Falta de replicação das modificações realizadas',
      'C13: Erro na definição da lógica do código',
      'C14: Erro na implementação de interface',
      'C15: Erro na implementação de parseamento de documentos',
      'C16: Falta de testes unitários na verificação do funcionamento do código',
      'C17: Falta de testes unitários na verificação do funcionamento do front',
      'C18: Falta de testes E2E para verificação da feature'
    ],
    'Concepção': [
        'C1: Falta de documentação ou documentação insuficiente',
        'C3: Requisito/regras de negócio insuficientes',
        'C4: Escopo não planejado',
        'C7: Falta de conhecimento sobre software de terceiro/externo'
    ],
    'Implantação': [
        'C6: Alteração externa não prevista (api ou lib externa modificada)'
    ]
}