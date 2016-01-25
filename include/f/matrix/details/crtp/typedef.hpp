#ifndef MSFDAIOJALSKDJHASLDKJHALKJFDHALKSJFHAIUH43O8YASFDJKSFDA84YHAJHALASD9UYH4
#define MSFDAIOJALSKDJHASLDKJHALKJFDHALKSJFHAIUH43O8YASFDJKSFDA84YHAJHALASD9UYH4

#include <f/stride_iterator/stride_iterator.hpp>

#include <cstddef>
#include <iterator>

namespace f
{

    template<typename Type, typename Allocator>
    struct crtp_typedef
    {
        typedef Type                                                        value_type;
        typedef value_type*                                                 iterator;
        typedef const value_type*                                           const_iterator;
        typedef Allocator                                                   allocator_type;
        typedef std::size_t                                                 size_type;
        typedef std::ptrdiff_t                                              difference_type;
        typedef typename allocator_type::pointer                            pointer;
        typedef typename allocator_type::const_pointer                      const_pointer;
        typedef stride_iterator<value_type*>                                matrix_stride_iterator;
        typedef stride_iterator<value_type*>                                row_type;
        typedef stride_iterator<const value_type*>                          const_row_type;
        typedef stride_iterator<value_type*>                                col_type;
        typedef stride_iterator<const value_type*>                          const_col_type;
        typedef stride_iterator<value_type*>                                diag_type;
        typedef stride_iterator<const value_type*>                          const_diag_type;
        typedef stride_iterator<value_type*>                                anti_diag_type;
        typedef stride_iterator<const value_type*>                          const_anti_diag_type;
        typedef std::reverse_iterator<iterator>                             reverse_iterator;
        typedef std::reverse_iterator<const_iterator>                       const_reverse_iterator;
        typedef std::reverse_iterator<matrix_stride_iterator>               reverse_matrix_stride_iterator;
        typedef std::reverse_iterator<row_type>                             reverse_row_type;
        typedef std::reverse_iterator<const_row_type>                       const_reverse_row_type;
        typedef std::reverse_iterator<col_type>                             reverse_col_type;
        typedef std::reverse_iterator<const_col_type>                       const_reverse_col_type;
        typedef std::reverse_iterator<diag_type>                            reverse_upper_diag_type;
        typedef std::reverse_iterator<const_diag_type>                      const_reverse_upper_diag_type;
        typedef std::reverse_iterator<diag_type>                            reverse_lower_diag_type;
        typedef std::reverse_iterator<const_diag_type>                      const_reverse_lower_diag_type;
        typedef std::reverse_iterator<diag_type>                            reverse_diag_type;
        typedef std::reverse_iterator<const_diag_type>                      const_reverse_diag_type;
        typedef std::reverse_iterator<anti_diag_type>                       reverse_anti_diag_type;
        typedef std::reverse_iterator<const_anti_diag_type>                 const_reverse_anti_diag_type;

    };//struct crtp_typedef

}//namespace f

#endif//_SFDAIOJALSKDJHASLDKJHALKJFDHALKSJFHAIUH43O8YASFDJKSFDA84YHAJHALASD9UYH4


